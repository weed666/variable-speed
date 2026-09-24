from __future__ import annotations

import collections
from enum import Enum

import numpy as np

from rl_train.envs.constant_acceleration_task import (
    AccelerationDeltaVTaskTupleSampler,
    ConstantAccelerationTask,
    ConstantAccelerationTaskSampler,
    build_acceleration_delta_v_task_tuples,
    build_constant_acceleration_tasks,
)
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation


class MyoAssistLegAcceleration(MyoAssistLegImitation):
    class Phase(Enum):
        INITIAL_HOLD = "INITIAL_HOLD"
        RAMP = "RAMP"
        FINAL_HOLD = "FINAL_HOLD"

    DEFAULT_OBS_KEYS = [
        "qpos",
        "qvel",
        "act",
        "sensor",
        "target_velocity",
        "target_acceleration",
        "goal_velocity",
    ]

    def _setup(
        self,
        *,
        env_params,
        reference_data: dict | None = None,
        **kwargs,
    ):
        self._stage3_reward_ready = False
        self._target_acceleration = 0.0
        self._goal_velocity = float(getattr(env_params, "min_target_velocity", 0.0))
        self._current_task = ConstantAccelerationTask(
            initial_velocity=self._goal_velocity,
            goal_velocity=self._goal_velocity,
            signed_acceleration=0.0,
            ramp_duration=1.0,
        )
        self._phase = self.Phase.INITIAL_HOLD
        self._phase_for_current_step = self.Phase.INITIAL_HOLD
        self._phase_transition_times = {}
        self._termination_reason = None
        self._fixed_evaluation_task = None
        self._fixed_reference_index = None
        self._ramp_start_time = 1e9
        self._ramp_end_time = 1e9 + 1.0
        self._episode_duration_s = float(getattr(env_params, "episode_duration_s", 8.0))
        self._configured_episode_duration_s = self._episode_duration_s
        self._reference_index_at_reset = 0
        self._stage3_reference_reset_max_index = int(round(3.0 * env_params.control_framerate))
        self._training_stage = str(getattr(env_params, "training_stage", "stage3_exo_teacher"))
        self._teacher_regime = str(getattr(env_params, "teacher_regime", "mixed"))
        self._task_pool_mode = str(getattr(env_params, "task_pool_mode", "speed_level_pairs"))
        self._dynamic_episode_duration = bool(getattr(env_params, "dynamic_episode_duration", False))
        self._speed_min = float(getattr(env_params, "speed_min", getattr(env_params, "min_target_velocity", 0.9)))
        self._speed_max = float(getattr(env_params, "speed_max", getattr(env_params, "max_target_velocity", 1.6)))
        self._pre_hold_min_s = float(getattr(env_params, "pre_hold_min_s", 1.0))
        self._pre_hold_max_s = float(getattr(env_params, "pre_hold_max_s", 2.0))
        self._post_hold_min_s = float(getattr(env_params, "post_hold_min_s", 1.0))
        self._post_hold_max_s = float(getattr(env_params, "post_hold_max_s", 2.0))
        self._steady_speed_sampling = str(getattr(env_params, "steady_speed_sampling", "uniform"))
        self._steady_episode_duration_min_s = float(getattr(env_params, "steady_episode_duration_min_s", 4.0))
        self._steady_episode_duration_max_s = float(getattr(env_params, "steady_episode_duration_max_s", 6.0))
        self._pre_hold_duration = 0.0
        self._post_hold_duration = 0.0
        self._delta_velocity = 0.0
        self._task_tuple_key = None

        self._validate_stage3_config(env_params)
        env_params.custom_max_episode_steps = int(round(env_params.episode_duration_s * env_params.control_framerate))

        tasks = build_constant_acceleration_tasks(
            speed_levels=env_params.speed_levels,
            acceleration_magnitudes=env_params.acceleration_magnitudes,
            min_ramp_duration=env_params.min_ramp_duration,
            max_ramp_duration=env_params.max_ramp_duration,
        )
        self._task_sampler = ConstantAccelerationTaskSampler(
            tasks,
            strategy=env_params.task_sampling_strategy,
            rng=self.np_random,
        )
        self._task_pool = tasks
        self._delta_v_task_tuple_pool = build_acceleration_delta_v_task_tuples(
            acceleration_magnitudes=env_params.acceleration_magnitudes,
            delta_velocity_levels=getattr(env_params, "delta_velocity_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            min_ramp_duration=env_params.min_ramp_duration,
            max_ramp_duration=env_params.max_ramp_duration,
        )
        self._delta_v_task_tuple_sampler = AccelerationDeltaVTaskTupleSampler(
            self._delta_v_task_tuple_pool,
            strategy=env_params.task_sampling_strategy,
            rng=self.np_random,
        )

        self._episode_duration_s = float(env_params.episode_duration_s)
        self._configured_episode_duration_s = self._episode_duration_s
        self._ramp_start_time_min = float(env_params.ramp_start_time_min)
        self._ramp_start_time_max = float(env_params.ramp_start_time_max)
        self._acceleration_window_duration = float(env_params.acceleration_window_duration)
        self._min_step_acceleration_interval = float(env_params.min_step_acceleration_interval)
        self._steady_forward_weight = float(env_params.steady_forward_weight)
        self._steady_average_velocity_weight = float(env_params.steady_average_velocity_weight)
        self._local_acceleration_weight = float(env_params.local_acceleration_weight)
        self._step_acceleration_weight = float(env_params.step_acceleration_weight)
        self._local_acceleration_error_scale = float(env_params.local_acceleration_error_scale)
        self._step_acceleration_error_scale = float(env_params.step_acceleration_error_scale)
        self._muscle_activation_weight = float(env_params.muscle_activation_weight)
        self._out_of_trajectory_threshold = float(getattr(env_params, "out_of_trajectory_threshold", 100.0))

        self._reset_stage3_episode_state()

        super()._setup(env_params=env_params, reference_data=reference_data, loop_reference_data=False, **kwargs)
        self._stage3_reward_ready = True

    def _validate_stage3_config(self, env_params) -> None:
        checks = {
            "episode_duration_s": getattr(env_params, "episode_duration_s", None),
            "ramp_start_time_min": getattr(env_params, "ramp_start_time_min", None),
            "ramp_start_time_max": getattr(env_params, "ramp_start_time_max", None),
            "min_ramp_duration": getattr(env_params, "min_ramp_duration", None),
            "max_ramp_duration": getattr(env_params, "max_ramp_duration", None),
            "acceleration_window_duration": getattr(env_params, "acceleration_window_duration", None),
            "min_step_acceleration_interval": getattr(env_params, "min_step_acceleration_interval", None),
        }
        for name, value in checks.items():
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if env_params.episode_duration_s <= 0:
            raise ValueError("episode_duration_s must be positive")
        if env_params.ramp_start_time_min < 0:
            raise ValueError("ramp_start_time_min must be non-negative")
        if env_params.ramp_start_time_min > env_params.ramp_start_time_max:
            raise ValueError("ramp_start_time_min must be <= ramp_start_time_max")
        if env_params.ramp_start_time_max + env_params.max_ramp_duration >= env_params.episode_duration_s:
            raise ValueError(
                "ramp_start_time_max + max_ramp_duration must be < episode_duration_s"
            )
        if getattr(env_params, "teacher_regime", "mixed") not in ("mixed", "steady", "acceleration", "deceleration"):
            raise ValueError("teacher_regime must be one of mixed, steady, acceleration, deceleration")
        if getattr(env_params, "task_pool_mode", "speed_level_pairs") not in ("speed_level_pairs", "acceleration_delta_v_grid"):
            raise ValueError("task_pool_mode must be speed_level_pairs or acceleration_delta_v_grid")
        if getattr(env_params, "pre_hold_min_s", 1.0) > getattr(env_params, "pre_hold_max_s", 2.0):
            raise ValueError("pre_hold_min_s must be <= pre_hold_max_s")
        if getattr(env_params, "post_hold_min_s", 1.0) > getattr(env_params, "post_hold_max_s", 2.0):
            raise ValueError("post_hold_min_s must be <= post_hold_max_s")
        if getattr(env_params, "steady_episode_duration_min_s", 4.0) > getattr(env_params, "steady_episode_duration_max_s", 6.0):
            raise ValueError("steady_episode_duration_min_s must be <= steady_episode_duration_max_s")
        if env_params.acceleration_window_duration <= 0:
            raise ValueError("acceleration_window_duration must be positive")
        if env_params.min_step_acceleration_interval <= 0:
            raise ValueError("min_step_acceleration_interval must be positive")
        for weight_name in (
            "steady_forward_weight",
            "steady_average_velocity_weight",
            "local_acceleration_weight",
            "step_acceleration_weight",
            "muscle_activation_weight",
        ):
            value = float(getattr(env_params, weight_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{weight_name} must be finite and non-negative")
        for scale_name in ("local_acceleration_error_scale", "step_acceleration_error_scale"):
            value = float(getattr(env_params, scale_name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{scale_name} must be finite and positive")

    def _reset_stage3_episode_state(self) -> None:
        self._phase = self.Phase.INITIAL_HOLD
        self._phase_for_current_step = self.Phase.INITIAL_HOLD
        self._target_acceleration = 0.0
        self._phase_transition_times = {self.Phase.INITIAL_HOLD.value: 0.0}
        self._acceleration_time_buffer: list[float] = []
        self._actual_velocity_buffer: list[float] = []
        self._estimated_acceleration = 0.0
        self._acceleration_window_sample_count = 0
        self._acceleration_window_start_time = 0.0
        self._step_average_acceleration = 0.0
        self._step_acceleration_interval_duration = 0.0
        self._step_acceleration_error = 0.0
        self._closed_by_heel_strike = False
        self._closed_by_ramp_end = False
        self._tail_interval_closed = False
        self._last_heel_strike_detected = False
        self._ramp_interval_start_time = None
        self._ramp_interval_start_velocity = None
        self._activation_cost_raw = 0.0
        self._episode_activation_integral = 0.0
        self._completed_ramp = False
        self._termination_reason = None
        self._last_local_acceleration_reward = 0.0
        self._last_step_acceleration_reward = 0.0
        self._pre_hold_duration = 0.0
        self._post_hold_duration = 0.0
        self._delta_velocity = 0.0
        self._task_tuple_key = None

    def set_fixed_evaluation_task(
        self,
        *,
        initial_velocity: float,
        goal_velocity: float,
        signed_acceleration: float,
        ramp_start_time: float,
        reference_index: int | None = None,
    ) -> None:
        signed_acceleration = float(signed_acceleration)
        delta_v = float(goal_velocity) - float(initial_velocity)
        if signed_acceleration == 0:
            if delta_v != 0:
                raise ValueError(
                    "signed_acceleration must be non-zero when initial_velocity and goal_velocity differ"
                )
            ramp_duration = 0.0
        else:
            if delta_v == 0:
                raise ValueError("initial_velocity and goal_velocity must differ")
            if np.sign(delta_v) != np.sign(signed_acceleration):
                raise ValueError("signed_acceleration sign must point from initial_velocity to goal_velocity")
            ramp_duration = abs(delta_v) / abs(signed_acceleration)
        self._fixed_evaluation_task = ConstantAccelerationTask(
            initial_velocity=float(initial_velocity),
            goal_velocity=float(goal_velocity),
            signed_acceleration=signed_acceleration,
            ramp_duration=float(ramp_duration),
        )
        self._fixed_ramp_start_time = float(ramp_start_time)
        self._fixed_reference_index = reference_index

    def clear_fixed_evaluation_task(self) -> None:
        self._fixed_evaluation_task = None
        self._fixed_reference_index = None

    def _select_stage3_reference_index(self, fixed_index: int | None = None) -> int:
        safe_max_index = self._reference_data_length - int(
            np.ceil((self._episode_duration_s + 1.2) / self.dt)
        )
        max_index = min(self._stage3_reference_reset_max_index, safe_max_index)
        if max_index < 0:
            raise ValueError(
                "reference data is too short for Stage 3 episode_duration_s "
                f"{self._episode_duration_s}"
            )
        if fixed_index is not None:
            index = int(fixed_index)
            if index < 0 or index > max_index:
                raise ValueError(
                    f"fixed Stage 3 reference index {index} is outside safe range [0, {max_index}]"
                )
            return index
        if not self._flag_random_ref_index:
            return 0
        return int(self.np_random.integers(0, max_index + 1))

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["target_acceleration"] = np.array([self._target_acceleration])
        obs_dict["goal_velocity"] = np.array([self._goal_velocity])
        return obs_dict

    def reset(self, **kwargs):
        self._reset_stage3_episode_state()

        if self._fixed_evaluation_task is not None:
            task = self._fixed_evaluation_task
            ramp_start_time = self._fixed_ramp_start_time
            self._episode_duration_s = self._configured_episode_duration_s
            self.CUSTOM_MAX_EPISODE_STEPS = int(round(self._episode_duration_s / self.dt))
            self._pre_hold_duration = float(ramp_start_time)
            self._post_hold_duration = max(0.0, self._episode_duration_s - ramp_start_time - task.ramp_duration)
        else:
            task, ramp_start_time = self._sample_training_task_and_timing()
        reference_index = self._select_stage3_reference_index(
            self._fixed_reference_index if self._fixed_evaluation_task is not None else None
        )

        self._current_task = task
        self._ramp_start_time = ramp_start_time
        self._ramp_end_time = self._ramp_start_time + task.ramp_duration
        self._target_velocity = task.initial_velocity
        self._target_acceleration = 0.0
        self._goal_velocity = task.goal_velocity
        self._reference_index_at_reset = int(reference_index)
        self._imitation_index = int(reference_index)
        self._velocity_mode_for_this_episode = MyoAssistLegBase.VelocityMode.CONSTANT_ACCELERATION
        self._phase = self.Phase.INITIAL_HOLD
        self._phase_for_current_step = self.Phase.INITIAL_HOLD

        self._follow_reference_motion(False)
        self.sim.data.joint("pelvis_tx").qvel[0] = task.initial_velocity

        obs = MyoAssistLegBase.reset(
            self,
            reset_qpos=self.sim.data.qpos.copy(),
            reset_qvel=self.sim.data.qvel.copy(),
            **kwargs,
        )
        return obs

    def _sample_training_task_and_timing(self) -> tuple[ConstantAccelerationTask, float]:
        if self._teacher_regime == "steady":
            target_velocity = float(self.np_random.uniform(self._speed_min, self._speed_max))
            episode_duration = float(
                self.np_random.uniform(
                    self._steady_episode_duration_min_s,
                    self._steady_episode_duration_max_s,
                )
            )
            self._episode_duration_s = episode_duration if self._dynamic_episode_duration else self._configured_episode_duration_s
            self.CUSTOM_MAX_EPISODE_STEPS = int(round(self._episode_duration_s / self.dt))
            self._pre_hold_duration = self._episode_duration_s
            self._post_hold_duration = 0.0
            self._delta_velocity = 0.0
            self._task_tuple_key = "steady"
            return (
                ConstantAccelerationTask(
                    initial_velocity=target_velocity,
                    goal_velocity=target_velocity,
                    signed_acceleration=0.0,
                    ramp_duration=0.0,
                    delta_velocity=0.0,
                    tuple_acceleration_magnitude=0.0,
                ),
                1e9,
            )

        if (
            self._task_pool_mode == "acceleration_delta_v_grid"
            and self._teacher_regime in ("acceleration", "deceleration")
        ):
            task_tuple = self._delta_v_task_tuple_sampler.sample_tuple()
            abs_acceleration = float(task_tuple.acceleration_magnitude)
            delta_velocity = float(task_tuple.delta_velocity)
            if self._teacher_regime == "acceleration":
                start_low = self._speed_min
                start_high = self._speed_max - delta_velocity
                signed_acceleration = abs_acceleration
            else:
                start_low = self._speed_min + delta_velocity
                start_high = self._speed_max
                signed_acceleration = -abs_acceleration
            if start_high + 1e-9 < start_low:
                raise ValueError(
                    "No legal start velocity for task tuple "
                    f"a={abs_acceleration}, delta_v={delta_velocity}, regime={self._teacher_regime}"
                )
            initial_velocity = float(self.np_random.uniform(start_low, start_high))
            goal_velocity = initial_velocity + np.sign(signed_acceleration) * delta_velocity
            if not (self._speed_min - 1e-9 <= initial_velocity <= self._speed_max + 1e-9):
                raise ValueError(f"sampled initial velocity outside speed range: {initial_velocity}")
            if not (self._speed_min - 1e-9 <= goal_velocity <= self._speed_max + 1e-9):
                raise ValueError(f"sampled goal velocity outside speed range: {goal_velocity}")

            pre_hold = float(self.np_random.uniform(self._pre_hold_min_s, self._pre_hold_max_s))
            post_hold = float(self.np_random.uniform(self._post_hold_min_s, self._post_hold_max_s))
            episode_duration = pre_hold + float(task_tuple.ramp_duration) + post_hold
            self._episode_duration_s = episode_duration if self._dynamic_episode_duration else self._configured_episode_duration_s
            self.CUSTOM_MAX_EPISODE_STEPS = int(round(self._episode_duration_s / self.dt))
            self._pre_hold_duration = pre_hold
            self._post_hold_duration = post_hold
            self._delta_velocity = delta_velocity
            self._task_tuple_key = f"{abs_acceleration:.6g}:{delta_velocity:.6g}"
            return (
                ConstantAccelerationTask(
                    initial_velocity=initial_velocity,
                    goal_velocity=float(goal_velocity),
                    signed_acceleration=float(signed_acceleration),
                    ramp_duration=float(task_tuple.ramp_duration),
                    delta_velocity=delta_velocity,
                    tuple_acceleration_magnitude=abs_acceleration,
                ),
                pre_hold,
            )

        task = self._task_sampler.sample()
        ramp_start_time = float(
            self.np_random.uniform(self._ramp_start_time_min, self._ramp_start_time_max)
        )
        self._episode_duration_s = self._configured_episode_duration_s
        self.CUSTOM_MAX_EPISODE_STEPS = int(round(self._episode_duration_s / self.dt))
        self._pre_hold_duration = ramp_start_time
        self._post_hold_duration = max(0.0, self._episode_duration_s - ramp_start_time - task.ramp_duration)
        self._delta_velocity = abs(task.goal_velocity - task.initial_velocity)
        self._task_tuple_key = f"{abs(task.signed_acceleration):.6g}:{self._delta_velocity:.6g}"
        return task, ramp_start_time

    def step(self, a, **kwargs):
        if not getattr(self, "_stage3_reward_ready", False):
            return MyoAssistLegBase.step(self, a, **kwargs)
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if truncated and self._termination_reason is None:
            if self._imitation_index >= self._reference_data_length - 1:
                self._termination_reason = "reference_end"
            else:
                self._termination_reason = "time_limit"
        info["termination_reason"] = self._termination_reason
        info["stage3"] = self.get_stage3_diagnostics()
        return (next_obs, reward, terminated, truncated, info)

    def _modulate_constant_acceleration_target_velocity(self) -> None:
        self._update_phase_and_command_for_time(float(self.sim.data.time))
        self._phase_for_current_step = self._phase

    def _update_phase_and_command_for_time(self, sim_time: float) -> None:
        task = self._current_task
        if sim_time < self._ramp_start_time:
            new_phase = self.Phase.INITIAL_HOLD
            target_velocity = task.initial_velocity
            target_acceleration = 0.0
        elif sim_time >= self._ramp_end_time:
            new_phase = self.Phase.FINAL_HOLD
            target_velocity = task.goal_velocity
            target_acceleration = 0.0
            self._completed_ramp = True
        else:
            new_phase = self.Phase.RAMP
            elapsed = sim_time - self._ramp_start_time
            target_velocity = task.initial_velocity + task.signed_acceleration * elapsed
            if task.signed_acceleration > 0:
                target_velocity = min(target_velocity, task.goal_velocity)
            else:
                target_velocity = max(target_velocity, task.goal_velocity)
            target_acceleration = task.signed_acceleration

        if new_phase != self._phase:
            self._phase_transition_times[new_phase.value] = sim_time
            if new_phase == self.Phase.RAMP:
                self._ramp_interval_start_time = sim_time
                self._ramp_interval_start_velocity = self._actual_pelvis_velocity()
        self._phase = new_phase
        self._target_velocity = float(target_velocity)
        self._target_acceleration = float(target_acceleration)
        self._goal_velocity = task.goal_velocity

    def get_reward_dict(self, obs_dict):
        if not getattr(self, "_stage3_reward_ready", False):
            return super().get_reward_dict(obs_dict)

        self._last_heel_strike_detected = False
        imitation_reward, _info = self._calculate_imitation_rewards(obs_dict)

        actual_velocity = self._actual_pelvis_velocity()
        sim_time = float(self.sim.data.time)
        local_reward = self._calculate_local_acceleration_reward(sim_time, actual_velocity)
        step_reward = self._calculate_step_acceleration_reward(sim_time, actual_velocity)

        muscle_activation = self._get_muscle_activation()
        self._activation_cost_raw = float(np.mean(muscle_activation))
        self._episode_activation_integral += self.dt * self._activation_cost_raw

        phase = self._phase_for_current_step
        if phase == self.Phase.RAMP:
            task_reward = (
                self._local_acceleration_weight * local_reward
                + self._step_acceleration_weight * step_reward
            )
            steady_velocity_reward = 0.0
            steady_average_velocity_reward = 0.0
        else:
            steady_velocity_reward = self._steady_forward_weight * imitation_reward["forward_reward"]
            steady_average_velocity_reward = (
                self._steady_average_velocity_weight * imitation_reward["average_velocity_per_step"]
            )
            task_reward = (
                steady_velocity_reward
                + steady_average_velocity_reward
            )
            if self._teacher_regime == "steady":
                task_reward += (
                    self._local_acceleration_weight * local_reward
                    + self._step_acceleration_weight * step_reward
                )
            else:
                local_reward = 0.0
                step_reward = 0.0

        imitation_weighted_reward = np.sum(
            [
                self.rwd_keys_wt.get(key, 0.0) * imitation_reward[key]
                for key in (
                    "qpos_imitation_rewards",
                    "qvel_imitation_rewards",
                    "end_effector_imitation_reward",
                )
                if key in imitation_reward
            ],
            axis=0,
        )
        dense_reward = (
            task_reward

            + self._muscle_activation_weight
            * imitation_reward["muscle_activation_penalty"]

            + self.rwd_keys_wt.get(
                "muscle_activation_diff_penalty", 0.0
            )
            * imitation_reward["muscle_activation_diff_penalty"]

            + self.rwd_keys_wt.get(
                "joint_constraint_force_penalty", 0.0
            )
            * imitation_reward["joint_constraint_force_penalty"]

            + self.rwd_keys_wt.get(
                "foot_force_penalty", 0.0
            )
            * imitation_reward["foot_force_penalty"]

            + imitation_weighted_reward
        )

        rwd_dict = collections.OrderedDict((key, imitation_reward[key]) for key in imitation_reward)
        rwd_dict.update({
            "local_acceleration_reward": float(local_reward),
            "step_average_acceleration_reward": float(step_reward),
            "steady_velocity_reward": float(steady_velocity_reward),
            "steady_average_velocity_reward": float(steady_average_velocity_reward),
            "stage3_imitation_reward": float(imitation_weighted_reward),
            "stage3_task_reward": float(task_reward),
            "activation_cost_raw": float(self._activation_cost_raw),
            "episode_activation_integral": float(self._episode_activation_integral),
            "sparse": 0,
            "solved": False,
            "done": self._get_done(),
        })
        rwd_dict["dense"] = float(dense_reward) if np.isfinite(dense_reward) else 0.0
        return rwd_dict

    def _calculate_local_acceleration_reward(self, sim_time: float, actual_velocity: float) -> float:
        self._acceleration_time_buffer.append(sim_time)
        self._actual_velocity_buffer.append(actual_velocity)
        min_time = sim_time - self._acceleration_window_duration
        keep = [idx for idx, value in enumerate(self._acceleration_time_buffer) if value >= min_time]
        self._acceleration_time_buffer = [self._acceleration_time_buffer[idx] for idx in keep]
        self._actual_velocity_buffer = [self._actual_velocity_buffer[idx] for idx in keep]

        self._acceleration_window_sample_count = len(self._acceleration_time_buffer)
        self._acceleration_window_start_time = (
            float(self._acceleration_time_buffer[0]) if self._acceleration_time_buffer else sim_time
        )
        if not self._is_acceleration_reward_active():
            self._estimated_acceleration = 0.0
            return 0.0
        if len(self._acceleration_time_buffer) < 2:
            self._estimated_acceleration = 0.0
            return 0.0

        times = np.array(self._acceleration_time_buffer, dtype=float)
        velocities = np.array(self._actual_velocity_buffer, dtype=float)
        t_centered = times - np.mean(times)
        denominator = float(np.sum(np.square(t_centered)))
        if denominator <= 1e-12:
            self._estimated_acceleration = 0.0
            return 0.0
        estimate = float(np.sum(t_centered * (velocities - np.mean(velocities))) / denominator)
        if not np.isfinite(estimate):
            self._estimated_acceleration = 0.0
            return 0.0
        self._estimated_acceleration = estimate
        error = estimate - self._current_task.signed_acceleration
        reward = self.dt * np.exp(-self._local_acceleration_error_scale * np.square(error))
        if not np.isfinite(reward):
            return 0.0
        self._last_local_acceleration_reward = float(reward)
        return float(reward)

    def _calculate_step_acceleration_reward(self, sim_time: float, actual_velocity: float) -> float:
        self._step_average_acceleration = 0.0
        self._step_acceleration_interval_duration = 0.0
        self._step_acceleration_error = 0.0
        self._closed_by_heel_strike = False
        self._closed_by_ramp_end = False

        if not self._is_acceleration_reward_active():
            return 0.0
        if self._ramp_interval_start_time is None:
            self._ramp_interval_start_time = max(self._ramp_start_time, sim_time)
            self._ramp_interval_start_velocity = actual_velocity
            return 0.0

        close_by_ramp_end = (
            self._phase_for_current_step == self.Phase.RAMP
            and sim_time >= self._ramp_end_time
            and not self._tail_interval_closed
        )
        close_by_heel = bool(self._last_heel_strike_detected)
        if not close_by_heel and not close_by_ramp_end:
            return 0.0

        reward = self._close_step_acceleration_interval(
            sim_time,
            actual_velocity,
            closed_by_heel_strike=close_by_heel,
            closed_by_ramp_end=close_by_ramp_end,
        )
        if close_by_ramp_end:
            self._tail_interval_closed = True
            self._completed_ramp = True
        return reward

    def _close_step_acceleration_interval(
        self,
        sim_time: float,
        actual_velocity: float,
        *,
        closed_by_heel_strike: bool,
        closed_by_ramp_end: bool,
    ) -> float:
        interval_duration = sim_time - float(self._ramp_interval_start_time)
        self._step_acceleration_interval_duration = float(interval_duration)
        self._closed_by_heel_strike = bool(closed_by_heel_strike)
        self._closed_by_ramp_end = bool(closed_by_ramp_end)
        if interval_duration < self._min_step_acceleration_interval:
            return 0.0

        step_average_acceleration = (
            actual_velocity - float(self._ramp_interval_start_velocity)
        ) / interval_duration
        if not np.isfinite(step_average_acceleration):
            return 0.0
        self._step_average_acceleration = float(step_average_acceleration)
        self._step_acceleration_error = float(
            step_average_acceleration - self._current_task.signed_acceleration
        )
        reward = self.dt * np.exp(
            -self._step_acceleration_error_scale * np.square(self._step_acceleration_error)
        )
        self._ramp_interval_start_time = sim_time
        self._ramp_interval_start_velocity = actual_velocity
        if not np.isfinite(reward):
            return 0.0
        self._last_step_acceleration_reward = float(reward)
        return float(reward)

    def _is_acceleration_reward_active(self) -> bool:
        return (
            self._phase_for_current_step == self.Phase.RAMP
            or (
                self._teacher_regime == "steady"
                and self._training_stage == "stage5_coadapt"
            )
        )

    def _detect_heel_strike(self):
        detected = super()._detect_heel_strike()
        self._last_heel_strike_detected = bool(detected)
        return detected

    def _actual_pelvis_velocity(self) -> float:
        return float(self.sim.data.joint("pelvis_tx").qvel[0].copy())

    def _get_done(self):
        if super()._get_done():
            self._termination_reason = "pelvis_height"
            return True
        if not np.all(np.isfinite(self.sim.data.qpos)) or not np.all(np.isfinite(self.sim.data.qvel)):
            self._termination_reason = "non_finite_state"
            return True
        return False

    def get_stage3_diagnostics(self) -> dict:
        return {
            "phase": self._phase_for_current_step.value,
            "target_velocity": float(self._target_velocity),
            "actual_pelvis_velocity": self._actual_pelvis_velocity(),
            "target_acceleration": float(self._target_acceleration),
            "estimated_acceleration": float(self._estimated_acceleration),
            "acceleration_window_sample_count": int(self._acceleration_window_sample_count),
            "acceleration_window_start_time": float(self._acceleration_window_start_time),
            "step_average_acceleration": float(self._step_average_acceleration),
            "interval_duration": float(self._step_acceleration_interval_duration),
            "step_acceleration_error": float(self._step_acceleration_error),
            "closed_by_heel_strike": bool(self._closed_by_heel_strike),
            "closed_by_ramp_end": bool(self._closed_by_ramp_end),
            "initial_velocity": float(self._current_task.initial_velocity),
            "goal_velocity": float(self._current_task.goal_velocity),
            "signed_acceleration": float(self._current_task.signed_acceleration),
            "abs_acceleration": float(abs(self._current_task.signed_acceleration)),
            "delta_velocity": float(self._delta_velocity),
            "ramp_start_time": float(self._ramp_start_time),
            "ramp_duration": float(self._current_task.ramp_duration),
            "ramp_end_time": float(self._ramp_end_time),
            "pre_hold_duration": float(self._pre_hold_duration),
            "post_hold_duration": float(self._post_hold_duration),
            "episode_duration": float(self._episode_duration_s),
            "teacher_regime": self._teacher_regime,
            "training_stage": self._training_stage,
            "task_pool_mode": self._task_pool_mode,
            "task_tuple_key": self._task_tuple_key,
            "reference_index": int(getattr(self, "_reference_index_at_reset", 0)),
            "phase_transition_times": dict(self._phase_transition_times),
            "activation_cost_raw": float(self._activation_cost_raw),
            "episode_activation_integral": float(self._episode_activation_integral),
            "completed_ramp": bool(self._completed_ramp),
            "termination_reason": self._termination_reason,
        }
