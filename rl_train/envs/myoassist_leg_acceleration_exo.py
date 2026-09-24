from __future__ import annotations

import collections
import math
import warnings

import numpy as np

from rl_train.envs.myoassist_leg_acceleration import MyoAssistLegAcceleration


class MyoAssistLegAccelerationExo(MyoAssistLegAcceleration):
    DEFAULT_OBS_KEYS = MyoAssistLegAcceleration.DEFAULT_OBS_KEYS + ["exo_action_history"]

    EXO_ACTUATOR_NAMES = ("Exo_R", "Exo_L")
    EXO_JOINT_NAMES = ("hip_flexion_r", "hip_flexion_l")
    HIP_REWARD_MUSCLE_NAMES = (
        "hamstrings_r",
        "glutmax_r",
        "iliopsoas_r",
        "rectfem_r",
        "hamstrings_l",
        "glutmax_l",
        "iliopsoas_l",
        "rectfem_l",
    )

    def _setup(
        self,
        *,
        env_params,
        reference_data: dict | None = None,
        **kwargs,
    ):
        self._exo_teacher_reward_ready = False
        self._exo_action_history = np.zeros(6, dtype=float)
        self._latest_normalized_exo_action = np.zeros(2, dtype=float)
        self._latest_raw_normalized_exo_action = np.zeros(2, dtype=float)
        self._exo_output_lpf_state = np.zeros(2, dtype=float)
        self._exo_output_lpf_enabled = bool(
            getattr(env_params, "exo_output_lpf_enabled", False)
        )
        tau_value = getattr(env_params, "exo_output_lpf_tau_s", None)
        self._exo_output_lpf_tau_s = None if tau_value is None or float(tau_value) <= 0.0 else float(tau_value)
        self._exo_output_lpf_alpha_effective = 0.0
        self._warn_or_reject_deprecated_lpf_alpha(env_params)
        self._stage5_observation_extension = bool(
            getattr(env_params, "stage5_observation_extension", False)
        )
        self._exo_torque_limit_nm = float(getattr(env_params, "exo_torque_limit_nm", 6.0))
        self._torque_limit_reference_nm = float(
            getattr(env_params, "torque_limit_reference_nm", 25.0)
        )
        self._stage5_smat_power_magnitude_saturation_reward_weight = float(
            getattr(env_params, "stage5_smat_power_magnitude_saturation_reward_weight", 0.0)
        )
        self._stage5_smat_torque_rate_penalty_weight = float(
            getattr(env_params, "stage5_smat_torque_rate_penalty_weight", 0.0)
        )
        self._stage5_smat_power_alpha = float(getattr(env_params, "stage5_smat_power_alpha", 0.3))
        self._stage5_smat_magnitude_beta = float(getattr(env_params, "stage5_smat_magnitude_beta", 0.15))
        self._stage5_smat_saturation_lambda = float(getattr(env_params, "stage5_smat_saturation_lambda", 2.0))
        self._stage5_smat_saturation_delta = float(getattr(env_params, "stage5_smat_saturation_delta", 0.8))
        self._stage5_smat_omega_scale_rad_s = float(getattr(env_params, "stage5_smat_omega_scale_rad_s", 2.0))
        self._hip_muscle_activation_reward_weight = float(
            getattr(env_params, "hip_muscle_activation_reward_weight", 2.0)
        )
        self._exo_power_reward_weight = float(
            getattr(env_params, "exo_power_reward_weight", 4.0)
        )
        self._exo_power_reward_alpha = float(
            getattr(env_params, "exo_power_reward_alpha", 0.5)
        )
        self._exo_power_reward_mode = str(
            getattr(env_params, "exo_power_reward_mode", "positive")
        )
        self._validate_exo_teacher_reward_config()
        if self._stage5_observation_extension:
            self.DEFAULT_OBS_KEYS = (
                MyoAssistLegAccelerationExo.DEFAULT_OBS_KEYS
                + ["stage5_applied_exo_command"]
            )

        super()._setup(env_params=env_params, reference_data=reference_data, **kwargs)
        self._setup_exo_output_lpf_from_dt()
        self._setup_exo_teacher_indices()
        self._apply_exo_torque_limit()
        self._exo_teacher_reward_ready = True

    def _warn_or_reject_deprecated_lpf_alpha(self, env_params) -> None:
        old_alpha = getattr(env_params, "exo_output_lpf_alpha", None)
        if old_alpha is None:
            return
        if self._exo_output_lpf_tau_s is None and self._exo_output_lpf_enabled:
            raise ValueError(
                "exo_output_lpf_alpha is deprecated and cannot configure an enabled LPF. "
                "Use exo_output_lpf_tau_s, for example exo_output_lpf_tau_s=0.1."
            )
        message = (
            "exo_output_lpf_alpha is deprecated and ignored. "
            "Use exo_output_lpf_tau_s for Park-style LPF parameterization."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    def _setup_exo_output_lpf_from_dt(self) -> None:
        if self._exo_output_lpf_tau_s is None:
            if self._exo_output_lpf_enabled:
                raise ValueError("exo_output_lpf_tau_s is required when exo_output_lpf_enabled is true")
            self._exo_output_lpf_tau_s = 0.1
        if not np.isfinite(self._exo_output_lpf_tau_s) or self._exo_output_lpf_tau_s <= 0.0:
            raise ValueError("exo_output_lpf_tau_s must be finite and positive")
        self._exo_output_lpf_alpha_effective = self.compute_exo_output_lpf_alpha(
            dt=float(self.dt),
            tau_s=self._exo_output_lpf_tau_s,
        )

    def _validate_exo_teacher_reward_config(self) -> None:
        if self._exo_torque_limit_nm <= 0 or not np.isfinite(self._exo_torque_limit_nm):
            raise ValueError("exo_torque_limit_nm must be finite and positive")
        if self._torque_limit_reference_nm <= 0 or not np.isfinite(self._torque_limit_reference_nm):
            raise ValueError("torque_limit_reference_nm must be finite and positive")
        if self._hip_muscle_activation_reward_weight < 0:
            raise ValueError("hip_muscle_activation_reward_weight must be non-negative")
        if self._exo_power_reward_weight < 0:
            raise ValueError("exo_power_reward_weight must be non-negative")
        if self._exo_power_reward_alpha < 0:
            raise ValueError("exo_power_reward_alpha must be non-negative")
        if self._exo_power_reward_mode not in ("positive", "phase_conditioned"):
            raise ValueError(
                "exo_power_reward_mode must be either 'positive' or 'phase_conditioned'"
            )
        for name, value in (
            ("stage5_smat_power_magnitude_saturation_reward_weight", self._stage5_smat_power_magnitude_saturation_reward_weight),
            ("stage5_smat_torque_rate_penalty_weight", self._stage5_smat_torque_rate_penalty_weight),
            ("stage5_smat_power_alpha", self._stage5_smat_power_alpha),
            ("stage5_smat_magnitude_beta", self._stage5_smat_magnitude_beta),
            ("stage5_smat_saturation_lambda", self._stage5_smat_saturation_lambda),
            ("stage5_smat_saturation_delta", self._stage5_smat_saturation_delta),
            ("stage5_smat_omega_scale_rad_s", self._stage5_smat_omega_scale_rad_s),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def _setup_exo_teacher_indices(self) -> None:
        self._exo_actuator_ids = [
            int(self.sim.model.actuator(name).id) for name in self.EXO_ACTUATOR_NAMES
        ]
        expected_ids = list(range(22, 24))
        if self._exo_actuator_ids != expected_ids:
            raise ValueError(
                f"expected Exo actuators at indices {expected_ids}, got {self._exo_actuator_ids}"
            )

        self._exo_joint_dofadr = []
        for joint_name in self.EXO_JOINT_NAMES:
            joint_id = int(self.sim.model.joint(joint_name).id)
            self._exo_joint_dofadr.append(int(self.sim.model.jnt_dofadr[joint_id]))

        self._hip_activation_indices = []
        self._hip_activation_name_to_index = collections.OrderedDict()
        for muscle_name in self.HIP_REWARD_MUSCLE_NAMES:
            actuator_id = int(self.sim.model.actuator(muscle_name).id)
            actadr = int(self.sim.model.actuator_actadr[actuator_id])
            actnum = int(self.sim.model.actuator_actnum[actuator_id])
            if actadr < 0 or actnum != 1:
                raise ValueError(f"{muscle_name} is not a single-state muscle actuator")
            self._hip_activation_indices.append(actadr)
            self._hip_activation_name_to_index[muscle_name] = actadr
        self._hip_activation_indices = np.array(self._hip_activation_indices, dtype=int)

    def _apply_exo_torque_limit(self) -> None:
        for actuator_name in self.EXO_ACTUATOR_NAMES:
            actuator_id = int(self.sim.model.actuator(actuator_name).id)
            self.sim.model.actuator_ctrlrange[actuator_id, 0] = -self._exo_torque_limit_nm
            self.sim.model.actuator_ctrlrange[actuator_id, 1] = self._exo_torque_limit_nm

    def _get_exo_torque_capacity_norm(self) -> float:
        return float(self._exo_torque_limit_nm / self._torque_limit_reference_nm)

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["exo_action_history"] = self._exo_action_history.copy()
        if self._stage5_observation_extension:
            obs_dict["stage5_applied_exo_command"] = self._latest_normalized_exo_action.copy()
        return obs_dict

    def reset(self, **kwargs):
        self._reset_exo_action_history()
        return super().reset(**kwargs)

    def step(self, a, **kwargs):
        action_to_execute = a
        raw_exo_action = self._extract_normalized_exo_action(a)
        executed_exo_action = raw_exo_action
        if self._exo_output_lpf_enabled and np.asarray(a).reshape(-1).size >= 24:
            executed_exo_action = self._apply_exo_output_lpf(raw_exo_action)
            action_to_execute = np.asarray(a, dtype=float).copy()
            action_to_execute[22:24] = executed_exo_action
        self._store_normalized_exo_action(raw_exo_action, executed_exo_action)
        next_obs, reward, terminated, truncated, info = super().step(action_to_execute, **kwargs)
        exo_info = self.get_exo_teacher_diagnostics()
        if isinstance(info, dict):
            stage3_info = info.get("stage3", {})
            if isinstance(stage3_info, dict):
                stage3_info.update(
                    {
                        "exo_torque_limit_nm": float(self._exo_torque_limit_nm),
                        "A_norm": float(self._get_exo_torque_capacity_norm()),
                    }
                )
            info["exo_teacher"] = exo_info
            teacher_task = {}
            if isinstance(stage3_info, dict):
                for key in (
                    "teacher_regime",
                    "training_stage",
                    "initial_velocity",
                    "goal_velocity",
                    "signed_acceleration",
                    "abs_acceleration",
                    "delta_velocity",
                    "ramp_duration",
                    "pre_hold_duration",
                    "post_hold_duration",
                    "episode_duration",
                    "phase",
                    "exo_torque_limit_nm",
                    "A_norm",
                    "task_tuple_key",
                ):
                    if key in stage3_info:
                        teacher_task[key] = stage3_info[key]
            info["teacher_task"] = teacher_task
        return next_obs, reward, terminated, truncated, info

    def _reset_exo_action_history(self) -> None:
        self._exo_action_history = np.zeros(6, dtype=float)
        self._latest_normalized_exo_action = np.zeros(2, dtype=float)
        self._latest_raw_normalized_exo_action = np.zeros(2, dtype=float)
        self._exo_output_lpf_state = np.zeros(2, dtype=float)

    @staticmethod
    def compute_exo_output_lpf_alpha(*, dt: float, tau_s: float) -> float:
        dt = float(dt)
        tau_s = float(tau_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("tau_s must be finite and positive")
        return float(1.0 - math.exp(-dt / tau_s))

    @staticmethod
    def compute_exo_output_lpf(raw_action: np.ndarray, previous_filtered_action: np.ndarray, alpha_paper: float) -> np.ndarray:
        raw = np.asarray(raw_action, dtype=float)
        previous = np.asarray(previous_filtered_action, dtype=float)
        return previous + float(alpha_paper) * (raw - previous)

    def _apply_exo_output_lpf(self, raw_exo_action: np.ndarray) -> np.ndarray:
        self._exo_output_lpf_state = self.compute_exo_output_lpf(
            raw_exo_action,
            self._exo_output_lpf_state,
            self._exo_output_lpf_alpha_effective,
        )
        return self._exo_output_lpf_state.copy()

    def _extract_normalized_exo_action(self, action) -> np.ndarray:
        action_array = np.asarray(action, dtype=float).reshape(-1)
        if action_array.size >= 24:
            return np.clip(action_array[22:24], -1.0, 1.0).astype(float)
        return np.zeros(2, dtype=float)

    def _store_normalized_exo_action(self, raw_exo_action, executed_exo_action) -> None:
        raw_exo_action = np.asarray(raw_exo_action, dtype=float)
        executed_exo_action = np.asarray(executed_exo_action, dtype=float)
        previous_history = self._exo_action_history.copy()
        self._latest_raw_normalized_exo_action = raw_exo_action.astype(float)
        self._latest_normalized_exo_action = executed_exo_action.astype(float)
        self._exo_action_history = np.concatenate(
            [self._latest_normalized_exo_action, previous_history[:4]]
        )

    def get_reward_dict(self, obs_dict):
        base_rwd = super().get_reward_dict(obs_dict)
        if not getattr(self, "_exo_teacher_reward_ready", False):
            return base_rwd

        hip_reward = self._calculate_hip_muscle_activation_reward()
        exo_power_reward = self._calculate_exo_power_reward()
        stage5_smat_reward = self._calculate_stage5_smat_power_magnitude_saturation_reward()
        stage5_torque_rate_penalty = self._calculate_stage5_smat_torque_rate_penalty()
        base_rwd["hip_muscle_activation_reward"] = float(hip_reward)
        base_rwd["exo_power_reward"] = float(exo_power_reward)
        base_rwd["stage5_smat_power_magnitude_saturation_reward"] = float(stage5_smat_reward)
        base_rwd["stage5_smat_torque_rate_penalty"] = float(stage5_torque_rate_penalty)
        dense = float(base_rwd.get("dense", 0.0))
        dense += self._hip_muscle_activation_reward_weight * hip_reward
        dense += self._exo_power_reward_weight * exo_power_reward
        dense += self._stage5_smat_power_magnitude_saturation_reward_weight * stage5_smat_reward
        dense += self._stage5_smat_torque_rate_penalty_weight * stage5_torque_rate_penalty
        base_rwd["dense"] = float(dense) if np.isfinite(dense) else 0.0
        return base_rwd

    def _calculate_hip_muscle_activation_reward(self) -> float:
        activations = self.sim.data.act[:].copy()[self._hip_activation_indices]
        reward = -self.dt * float(np.mean(np.square(activations)))
        return reward if np.isfinite(reward) else 0.0

    def _calculate_exo_power_reward(self) -> float:
        power = self._get_exo_power()
        desired_power_sign = self._get_desired_power_sign()
        signed_power = desired_power_sign * power
        executed_exo_action = self._get_latest_executed_normalized_exo_action()
        reward = self.dt * float(
            np.sum(
                self._exo_power_reward_alpha
                * np.square(executed_exo_action)
                * np.sign(signed_power)
            )
        )
        return reward if np.isfinite(reward) else 0.0

    def _get_latest_executed_normalized_exo_action(self) -> np.ndarray:
        return np.asarray(self._latest_normalized_exo_action, dtype=float)

    @staticmethod
    def compute_stage5_smat_power_magnitude_saturation_reward(
        *,
        dt: float,
        normalized_exo_action: np.ndarray,
        hip_qvel: np.ndarray,
        omega_scale: float,
        alpha: float,
        beta: float,
        saturation_lambda: float,
        saturation_delta: float,
    ) -> float:
        u_hat = np.asarray(normalized_exo_action, dtype=float)
        omega_hat = np.clip(np.asarray(hip_qvel, dtype=float) / float(omega_scale), -1.0, 1.0)
        saturation = np.maximum(0.0, np.abs(u_hat) - float(saturation_delta))
        reward = float(dt) * float(
            np.sum(
                float(alpha) * u_hat * omega_hat
                - float(beta) * np.square(u_hat)
                - float(saturation_lambda) * np.square(saturation)
            )
        )
        return reward if np.isfinite(reward) else 0.0

    @staticmethod
    def compute_stage5_smat_torque_rate_penalty(
        *,
        dt: float,
        normalized_exo_action: np.ndarray,
        previous_normalized_exo_action: np.ndarray,
    ) -> float:
        delta = np.asarray(normalized_exo_action, dtype=float) - np.asarray(
            previous_normalized_exo_action, dtype=float
        )
        penalty = -float(dt) * float(np.sum(np.square(delta)))
        return penalty if np.isfinite(penalty) else 0.0

    def _calculate_stage5_smat_power_magnitude_saturation_reward(self) -> float:
        hip_qvel = np.array(
            [
                float(self.sim.data.joint(joint_name).qvel[0].copy())
                for joint_name in self.EXO_JOINT_NAMES
            ],
            dtype=float,
        )
        executed_exo_action = self._get_latest_executed_normalized_exo_action()
        return self.compute_stage5_smat_power_magnitude_saturation_reward(
            dt=self.dt,
            normalized_exo_action=executed_exo_action,
            hip_qvel=hip_qvel,
            omega_scale=self._stage5_smat_omega_scale_rad_s,
            alpha=self._stage5_smat_power_alpha,
            beta=self._stage5_smat_magnitude_beta,
            saturation_lambda=self._stage5_smat_saturation_lambda,
            saturation_delta=self._stage5_smat_saturation_delta,
        )

    def _calculate_stage5_smat_torque_rate_penalty(self) -> float:
        previous_action = np.asarray(self._exo_action_history[2:4], dtype=float)
        executed_exo_action = self._get_latest_executed_normalized_exo_action()
        return self.compute_stage5_smat_torque_rate_penalty(
            dt=self.dt,
            normalized_exo_action=executed_exo_action,
            previous_normalized_exo_action=previous_action,
        )

    def _get_desired_power_sign(self) -> float:
        if self._exo_power_reward_mode == "positive":
            return 1.0
        if self._phase_for_current_step != self.Phase.RAMP:
            return 1.0
        signed_acceleration = float(self._current_task.signed_acceleration)
        if signed_acceleration < 0:
            return -1.0
        return 1.0

    def _get_exo_joint_torque(self) -> np.ndarray:
        torques = []
        for actuator_id, dofadr, actuator_name in zip(
            self._exo_actuator_ids,
            self._exo_joint_dofadr,
            self.EXO_ACTUATOR_NAMES,
        ):
            actuator_force = float(self.sim.data.actuator(actuator_name).force[0].copy())
            try:
                moment = float(self.sim.data.actuator_moment[actuator_id, dofadr])
            except Exception:
                moment = float(self.sim.model.actuator_gear[actuator_id, 0])
            torques.append(actuator_force * moment)
        return np.array(torques, dtype=float)

    def _get_exo_power(self) -> np.ndarray:
        hip_qvel = np.array(
            [
                float(self.sim.data.joint(joint_name).qvel[0].copy())
                for joint_name in self.EXO_JOINT_NAMES
            ],
            dtype=float,
        )
        return self._get_exo_joint_torque() * hip_qvel

    def get_exo_teacher_diagnostics(self) -> dict:
        return {
            "exo_action_history": self._exo_action_history.copy().tolist(),
            "latest_normalized_exo_action": self._latest_normalized_exo_action.copy().tolist(),
            "latest_raw_normalized_exo_action": self._latest_raw_normalized_exo_action.copy().tolist(),
            "exo_output_lpf_enabled": bool(self._exo_output_lpf_enabled),
            "exo_output_lpf_tau_s": float(self._exo_output_lpf_tau_s),
            "exo_output_lpf_alpha_effective": float(self._exo_output_lpf_alpha_effective),
            "exo_output_lpf_alpha_convention": "paper: u_filt = u_prev + alpha*(u_raw-u_prev)",
            "exo_output_lpf_state": self._exo_output_lpf_state.copy().tolist(),
            "exo_torque_limit_nm": float(self._exo_torque_limit_nm),
            "torque_limit_reference_nm": float(self._torque_limit_reference_nm),
            "A_norm": float(self._get_exo_torque_capacity_norm()),
            "exo_joint_torque": self._get_exo_joint_torque().tolist()
            if getattr(self, "_exo_teacher_reward_ready", False)
            else [0.0, 0.0],
            "exo_power": self._get_exo_power().tolist()
            if getattr(self, "_exo_teacher_reward_ready", False)
            else [0.0, 0.0],
            "desired_power_sign": self._get_desired_power_sign()
            if getattr(self, "_exo_teacher_reward_ready", False)
            else 1.0,
            "hip_activation_indices": dict(getattr(self, "_hip_activation_name_to_index", {})),
        }
