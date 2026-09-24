import numpy as np
from dataclasses import dataclass
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from myosuite.utils import gym
from myosuite.envs import env_base
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
import collections
import mujoco
from myoassist_utils.hfield_manager import HfieldManager
from enum import Enum
import random
import numpy as np


class MyoAssistLegBase(env_base.MujocoEnv):
    MYO_CREDIT = """\
    NeuMove MyoLegBase
    """
    MIN_HEEL_OFF_STANCE_RATIO = 0.425
    UNIFORM_CYCLE_TARGET_VELOCITIES = (
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
    )

    class VelocityMode(Enum):
        UNIFORM = 0
        SINUSOIDAL = 1
        STEP = 2
        CONSTANT_ACCELERATION = 3

    LEGACY_RANDOM_VELOCITY_MODES = (
        VelocityMode.UNIFORM,
        VelocityMode.SINUSOIDAL,
        VelocityMode.STEP,
    )

    DEFAULT_OBS_KEYS = ['qpos',
                        'qvel',
                        'act',
                        'sensor',
                        'target_velocity',
                        ]
    
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        print(f"=================environment seed: {seed}=====================")
        print(f"=================environment model_path: {model_path}=====================")
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self,*,
               env_params:TrainSessionConfigBase.EnvParams,
                **kwargs):
        
        self.is_evaluate_mode = kwargs.pop("is_evaluate_mode", False)

        self.sim.model.opt.timestep = 1 / env_params.physics_sim_framerate
        self._safe_height = env_params.safe_height

        self._min_target_velocity = env_params.min_target_velocity
        self._max_target_velocity = env_params.max_target_velocity
        self._configured_velocity_mode = self._parse_configured_velocity_mode(
            getattr(env_params, "velocity_mode", None)
        )
        self._uniform_sampling_strategy = getattr(env_params, "uniform_sampling_strategy", "random")
        self._validate_uniform_sampling_config()
        self._uniform_cycle_velocity_index = 0
        self._uniform_cycle_velocity_direction = 1
        self._min_target_velocity_period = env_params.min_target_velocity_period
        self._max_target_velocity_period = env_params.max_target_velocity_period
        self._change_mode_and_target_velocity_randomly()

        self._step_count_per_episode = 0
        self.CUSTOM_MAX_EPISODE_STEPS = env_params.custom_max_episode_steps

        self._prev_muscle_activations_for_reward = None

        self._enable_lumbar_joint = env_params.enable_lumbar_joint
        self._lumbar_joint_fixed_angle = env_params.lumbar_joint_fixed_angle
        self._lumbar_joint_damping_value = env_params.lumbar_joint_damping_value

        self.observation_joint_pos_keys = env_params.observation_joint_pos_keys
        self.observation_joint_vel_keys = env_params.observation_joint_vel_keys
        self.observation_sensor_keys = env_params.observation_sensor_keys

        self.joint_limit_sensor_keys = env_params.joint_limit_sensor_keys

        # Safely check whether the joint named "lumbar_extension" exists in the model.
        try:
            lumbar_joint_id = self.sim.model.joint("lumbar_extension").id  # Raises if joint is absent
            has_lumbar_extension = True
        except (KeyError, ValueError, TypeError):
            has_lumbar_extension = False

        if not self._enable_lumbar_joint:
            if 'lumbar_extension' in self.observation_joint_pos_keys:
                self.observation_joint_pos_keys.remove('lumbar_extension')
            if 'lumbar_extension' in self.observation_joint_vel_keys:
                self.observation_joint_vel_keys.remove('lumbar_extension')
            if has_lumbar_extension:
                # Fix the lumbar joint to a constant position and (optionally) remove it from observations
                self.sim.data.joint("lumbar_extension").qpos[0] = self._lumbar_joint_fixed_angle
                self.sim.model.jnt_range[lumbar_joint_id] = [
                    self._lumbar_joint_fixed_angle,
                    self._lumbar_joint_fixed_angle + 1e-6,
                ]


                # Adjust damping (whether the joint is fixed or not)
                dof_adr = self.sim.model.jnt_dofadr[lumbar_joint_id]
                joint_type = self.sim.model.jnt_type[lumbar_joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    dof_count = 6
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dof_count = 3
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    dof_count = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dof_count = 1
                else:
                    dof_count = 0  # Currently unused

                self.sim.model.dof_damping[dof_adr] = self._lumbar_joint_damping_value
            else:
                self.sim.model.body("torso").quat = [1, 0, 0, self._lumbar_joint_fixed_angle]
        
        
        #phys: 1000hz
        # control 50hz : 50 * 20 = 1000hz
        # ref 50hz: 500hz 10skip: 20 * 500 / 1000

        frame_skip = env_params.physics_sim_framerate // env_params.control_framerate
        original_reward_dict = DictionableDataclass.to_dict(env_params.reward_keys_and_weights)
        self.rwd_keys_wt = {}
        for key, value in original_reward_dict.items():
            if type(value) == dict:
                weight_sum = sum(value.values())
                self.rwd_keys_wt[key] = weight_sum
            else:
                self.rwd_keys_wt[key] = value

        self._initialize_pose()

        # reward per step
        self._muscle_activation_symmetry_pairs = self._build_muscle_activation_symmetry_pairs()
        self._reset_heel_strike_buffer()
        self._reset_gait_symmetry_reward_state()
        self._reset_muscle_activation_symmetry_state()
        self._reset_rearfoot_participation_state()
        self._reset_pathological_foot_contact_state()
        self._reset_reward_per_step()
        self._reset_properties_per_step()

        
        
        # self.renderer = self.sim._create_renderer(self.sim)



        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                weighted_reward_keys=self.rwd_keys_wt,
                frame_skip=frame_skip,
                **kwargs,
                )
        
        # Check if the keys in DEFAULT_OBS_KEYS are in the keys of the observation dictionary
        obs_dict_keys = list(self.get_obs_dict(self.sim).keys())
        assert (set(self.DEFAULT_OBS_KEYS + ['time'])) == set(obs_dict_keys), f"DEFAULT_OBS_KEYS != get_obs_dict.keys. DEFAULT_OBS_KEYS: {self.DEFAULT_OBS_KEYS}, get_obs_dict keys: {obs_dict_keys}"
        actual_reward_keys = list(self.get_reward_dict(self.sim).keys())
        assert (set(list(self.rwd_keys_wt.keys()) + ['dense', 'sparse', 'solved', 'done'])) == set(actual_reward_keys), f"rwd_keys_wt != actual_reward_keys. rwd_keys_wt: {self.rwd_keys_wt}, actual_reward_keys keys: {actual_reward_keys}"
        
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        # move heightfield down if not used
        # self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        # self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

        self._terrain_type = env_params.terrain_type
        self._terrain_params = env_params.terrain_params
        self._hfield_manager = HfieldManager(self.sim, "terrain", self.np_random)
        self._hfield_manager.set_hfield(self._terrain_type, self._terrain_params)

        observation, _reward, done, *_, _info = self.step(np.zeros(self.sim.model.nu))
        # if qpos set to all zero, joint looks weird, 30 steps will make it normal
        for _ in range(30):
            super().step(a=np.zeros(self.sim.model.nu))

    # override from MujocoEnv
    def get_obs_dict(self, sim):
        # TODO observation - tx exclude
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time]) # they use time separately like t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)

        qpos = []
        for key in self.observation_joint_pos_keys:
            qpos.append(sim.data.joint(f"{key}").qpos[0].copy())
        qvel = []
        for key in self.observation_joint_vel_keys:
            qvel.append(sim.data.joint(f"{key}").qvel[0].copy())
        obs_dict['qpos'] = np.array(qpos) # 7 + 1 elements
        obs_dict['qvel'] = np.array(qvel) # 7 + 2 elements
        if sim.model.na>0:
            # BaseV0 Add the key like this: obs_keys.append("act")
            obs_dict['act'] = sim.data.act[:].copy() # 22 elements
        obs_dict['sensor'] = []
        for key in self.observation_sensor_keys:
            sensor_data = sim.data.sensor(f"{key}").data.copy()
            if "foot" in key or "toes" in key:
                model_mass = np.sum(self.sim.model.body_mass)
                sensor_data = sensor_data / (model_mass * 9.81)
            obs_dict['sensor'].extend(sensor_data)
        obs_dict['sensor'] = np.array(obs_dict['sensor'])

        obs_dict['target_velocity'] = np.array([self._target_velocity])

        return obs_dict
    def _calculate_reward_per_step(self, obs_dict, muscle_activations):
        self._footstep_delta_time += self.dt
        self._delta_velocity_sum += self.dt * (self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity)
        self._activation_square_sum += np.sum(np.square(muscle_activations)) * self.dt
        
        time_passed = self.sim.data.time - self._prev_step_time
        if self._detect_heel_strike() and time_passed > 0:
            travel_distance = self.sim.data.body('pelvis').xpos[0] - self._prev_pelvis_tx_pos
            self.reward_muscle_activation_penalty_per_step = self.dt * (-self._activation_square_sum)
            leg_length = 1 # see https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py
            self.reward_average_velocity_per_step = self.dt * (-np.abs(self._delta_velocity_sum)) / leg_length
            self.reward_footstep_delta_time = self.dt * self._footstep_delta_time

            self._reset_properties_per_step()
        else:
            pass
            # reward_muscle_activation_penalty_per_step = 0.0
            # reward_average_velocity_per_step = 0.0
            # reward_footstep_delta_time = 0.0
        reward_per_steps = {
            'muscle_activation_penalty_per_step': float(self.reward_muscle_activation_penalty_per_step),
            'average_velocity_per_step': float(self.reward_average_velocity_per_step),
            'footstep_delta_time': float(self.reward_footstep_delta_time),
        }
        info = {}
        return reward_per_steps, info
    def _calculate_base_reward(self, obs_dict):
        model_mass = np.sum(self.sim.model.body_mass)
        # print(f"DEBUG:: model_mass: {model_mass}")
        model_weight = model_mass * 9.81 # in Newtons

        forward_reward = self.dt * np.exp(-5 * np.square(self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity))

        muscle_activations = self._get_muscle_activation()
        muscle_activation_penalty = - self.dt * np.mean(muscle_activations)

        joint_constraint_force_penalty = - self.dt * self._get_max_joint_constraint_force() / (model_weight)

        # TODO: take off muscle activation penalty from imitation rewards
        # Gait symmetry reward: infer whether the existing heel-strike logic fired without changing it.
        symmetry_prev_step_time = self._prev_step_time
        reward_per_steps, info = self._calculate_reward_per_step(obs_dict, muscle_activations)
        symmetry_heel_strike_detected = self._prev_step_time != symmetry_prev_step_time
        self._update_gait_symmetry_contact_state()
        gait_symmetry_reward = self._get_gait_symmetry_reward(symmetry_heel_strike_detected)
        symmetry_step_side = self._last_heel_strike_foot if symmetry_heel_strike_detected else None
        muscle_activation_symmetry_penalty = self._update_muscle_activation_symmetry_penalty(
            muscle_activations,
            symmetry_step_side,
        )
        rearfoot_participation_penalty = self._calculate_rearfoot_participation_penalty()
        (
            toe_first_penalty,
            rearfoot_persistence_penalty,
            premature_heel_off_penalty,
        ) = self._calculate_pathological_foot_contact_penalties()

        if self._prev_muscle_activations_for_reward is not None:
            muscle_activation_diff_penalty = self.dt * np.mean(np.exp(-4 * np.square(self._prev_muscle_activations_for_reward - muscle_activations)))
        else:
            muscle_activation_diff_penalty = 0
        self._prev_muscle_activations_for_reward = muscle_activations

        
        normalized_foot_force_sum = (np.abs(self._get_foot_force('r')) + np.abs(self._get_foot_force('l'))) / model_weight
        # print(f"DEBUG:: normalized_foot_force_sum: {normalized_foot_force_sum}")
        # e^(-max(0, f/w - 1))
        # foot_force_penalty = self.dt * np.exp(-np.maximum(0, normalized_foot_force_sum - 1))
        foot_force_penalty = -self.dt * max(
            0.0,
            normalized_foot_force_sum - 1.2,
        )
        # foot_force_penalty = self.dt * max(0, 1.2 - normalized_foot_force_sum)
        # print(f"DEBUG:: foot_force_penalty: {foot_force_penalty}")

        pelvis_height = self.sim.data.joint("pelvis_ty").qpos[0].copy()
        warning_height = 0.80
        fall_height = self._safe_height
        fall_shaping_k = 0.5
        terminal_fall_penalty = -1.0
        if pelvis_height >= warning_height:
            height_penalty = 0.0
        else:
            z = np.clip(
                (warning_height - pelvis_height) / (warning_height - fall_height),
                0.0,
                1.0,
            )
            height_penalty = -fall_shaping_k * self.dt * z**2
        is_fall = pelvis_height < self._safe_height
        fall_stability_reward = height_penalty + (terminal_fall_penalty if is_fall else 0.0)

        base_reward = {
            'forward_reward': forward_reward,
            'muscle_activation_penalty': muscle_activation_penalty,
            'muscle_activation_diff_penalty': muscle_activation_diff_penalty,
            'foot_force_penalty': foot_force_penalty,
            'joint_constraint_force_penalty': joint_constraint_force_penalty,
            'gait_symmetry_reward': gait_symmetry_reward,
            'muscle_activation_symmetry_penalty': muscle_activation_symmetry_penalty,
            'rearfoot_participation_penalty': rearfoot_participation_penalty,
            'toe_first_penalty': toe_first_penalty,
            'rearfoot_persistence_penalty': rearfoot_persistence_penalty,
            'premature_heel_off_penalty': premature_heel_off_penalty,
            'fall_stability_reward': fall_stability_reward,
        }
        # Update base_reward with reward_per_steps
        base_reward.update(reward_per_steps)

        info = {
            "muscle_activations": muscle_activations,
        }
        return base_reward, info
    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):

        base_reward, info = self._calculate_base_reward(obs_dict)

        # Automatically add all base_reward items to rwd_dict
        rwd_dict = collections.OrderedDict((key, base_reward[key]) for key in base_reward)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),  # env will use this to determine if the episode is over (see _forward in env_base.py)
        })
        # rwd_keys_wt: from MujocoEnv
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def step(self, a, **kwargs):
        self._modulate_target_velocity()
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        self._step_count_per_episode += 1
        is_over_time_limit = self._step_count_per_episode >= self.CUSTOM_MAX_EPISODE_STEPS
        
        return (next_obs, reward, terminated, truncated or is_over_time_limit, info)
    def just_forward(self):
        self.sim.forward()
    def set_target_velocity_mode_manually(self, mode:VelocityMode,
                                          starting_phase:float,
                                          initial_target_velocity:float,
                                          min_target_velocity:float,
                                          max_target_velocity:float,
                                          target_velocity_period:float = None):
        self._velocity_mode_for_this_episode = mode
        self._starting_phase = starting_phase
        if mode == MyoAssistLegBase.VelocityMode.SINUSOIDAL and target_velocity_period is None:
            raise ValueError("target_velocity_period must be provided for sinusoidal mode")
        self._target_velocity_period = target_velocity_period
        # self._modulate_target_velocity()
        self._target_velocity = initial_target_velocity
        self._prev_step_changed_time = self.sim.data.time

        self._min_target_velocity = min_target_velocity
        self._max_target_velocity = max_target_velocity
    def _change_mode_and_target_velocity_randomly(self):
        if self._configured_velocity_mode is None:
            velocity_mode_for_this_episode = random.choice(MyoAssistLegBase.LEGACY_RANDOM_VELOCITY_MODES)
        else:
            velocity_mode_for_this_episode = self._configured_velocity_mode
        starting_phase = random.uniform(0, 2 * np.pi)
        target_velocity_period = random.uniform(self._min_target_velocity_period, self._max_target_velocity_period) # maximum acc/dec is self._target_velocity_period / 2
        self.set_target_velocity_mode_manually(velocity_mode_for_this_episode,
            starting_phase=starting_phase,
            initial_target_velocity=self._min_target_velocity,
            min_target_velocity=self._min_target_velocity,
            max_target_velocity=self._max_target_velocity,
            target_velocity_period=target_velocity_period,)
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            self._target_velocity = self._sample_uniform_target_velocity()
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)

    def _parse_configured_velocity_mode(self, velocity_mode):
        if velocity_mode is None:
            return None
        try:
            return MyoAssistLegBase.VelocityMode[velocity_mode]
        except KeyError:
            allowed_values = ", ".join(mode.name for mode in MyoAssistLegBase.VelocityMode)
            raise ValueError(
                f"velocity_mode must be one of: {allowed_values}, or null; "
                f"got {velocity_mode!r}"
            )

    def _validate_uniform_sampling_config(self):
        if self._uniform_sampling_strategy not in ("random", "cycle"):
            raise ValueError(
                "uniform_sampling_strategy must be 'random' or 'cycle', "
                f"got {self._uniform_sampling_strategy!r}"
            )
        cycle_velocities = np.array(self.UNIFORM_CYCLE_TARGET_VELOCITIES, dtype=float)
        if len(cycle_velocities) < 2:
            raise ValueError("UNIFORM_CYCLE_TARGET_VELOCITIES must contain at least two velocities")
        if not np.all(np.isfinite(cycle_velocities)):
            raise ValueError("UNIFORM_CYCLE_TARGET_VELOCITIES must contain only finite values")
        if len(np.unique(cycle_velocities)) != len(cycle_velocities):
            raise ValueError("UNIFORM_CYCLE_TARGET_VELOCITIES must not contain duplicates")
        if not np.all(np.diff(cycle_velocities) > 0):
            raise ValueError("UNIFORM_CYCLE_TARGET_VELOCITIES must be strictly increasing")

    def _sample_uniform_target_velocity(self):
        if self._uniform_sampling_strategy == "random":
            return random.uniform(self._min_target_velocity, self._max_target_velocity)
        velocity = self.UNIFORM_CYCLE_TARGET_VELOCITIES[self._uniform_cycle_velocity_index]
        self._advance_uniform_cycle_velocity()
        return velocity

    def _advance_uniform_cycle_velocity(self):
        next_index = self._uniform_cycle_velocity_index + self._uniform_cycle_velocity_direction
        if next_index >= len(self.UNIFORM_CYCLE_TARGET_VELOCITIES):
            self._uniform_cycle_velocity_direction = -1
            next_index = len(self.UNIFORM_CYCLE_TARGET_VELOCITIES) - 2
        elif next_index < 0:
            self._uniform_cycle_velocity_direction = 1
            next_index = 1
        self._uniform_cycle_velocity_index = next_index

    def _calc_sinusoidal_target_velocity(self, phase:float, period:float, min_velocity:float, max_velocity:float):
        return min_velocity\
            + (max_velocity - min_velocity)\
                  * (np.sin(phase + 2 * np.pi * self.sim.data.time / (period)) + 1) / 2
    def _modulate_target_velocity(self):
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            # print(f"DEBUG:: {self.is_evaluate_mode} (mode:{self._velocity_mode_for_this_episode}, starting_phase:{self._starting_phase}, target_velocity_period:{self._target_velocity_period})")
            pass
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            if self.sim.data.time - self._prev_step_changed_time > self._target_velocity_period:
                self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)
                self._prev_step_changed_time = self.sim.data.time
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.CONSTANT_ACCELERATION:
            self._modulate_constant_acceleration_target_velocity()
    def reset(self, **kwargs):
        self._step_count_per_episode = 0
        if (
            not self.is_evaluate_mode
            and self._velocity_mode_for_this_episode != MyoAssistLegBase.VelocityMode.CONSTANT_ACCELERATION
        ):
            self._change_mode_and_target_velocity_randomly()
        self.sim.data.joint("pelvis_tx").qvel[0] = self._target_velocity

        self.sim.forward()
        # sync targets to sim_obsd
        self.robot.sync_sims(self.sim, self.sim_obsd)

        self._reset_heel_strike_buffer()
        self._reset_gait_symmetry_reward_state()
        self._reset_muscle_activation_symmetry_state()
        self._reset_rearfoot_participation_state()
        self._reset_pathological_foot_contact_state()
        self._reset_reward_per_step()
        self._reset_properties_per_step()

        # generate resets
        # obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        obs = super().reset(**kwargs)
        return obs
    
    def _get_done(self):
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self._safe_height:
            return True
        return False
    
    def _get_muscle_activation(self):
        if not self._enable_lumbar_joint:
            return self.sim.data.act[:].copy()
        muscle_activations_with_lumbar = np.concatenate((self.sim.data.act[:].copy(), 
                                              np.array([self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()]).reshape(1,)))
        return muscle_activations_with_lumbar

    def _build_muscle_activation_symmetry_pairs(self):
        """Pair left/right human muscle activations from MuJoCo actuator names."""
        actuator_by_base_and_side = {}
        for actuator_id in range(self.sim.model.nu):
            actuator_name = self.sim.model.id2name(actuator_id, "actuator")
            if actuator_name is None:
                continue
            if not (actuator_name.endswith("_l") or actuator_name.endswith("_r")):
                continue
            if "exo" in actuator_name.lower():
                continue
            act_adr = int(self.sim.model.actuator_actadr[actuator_id])
            act_num = int(self.sim.model.actuator_actnum[actuator_id])
            if act_adr < 0 or act_num != 1:
                continue

            base_name = actuator_name[:-2]
            side = actuator_name[-1]
            actuator_by_base_and_side.setdefault(base_name, {})[side] = {
                "name": actuator_name,
                "actuator_id": actuator_id,
                "activation_index": act_adr,
            }

        pairs = []
        incomplete_pairs = {}
        for base_name in sorted(actuator_by_base_and_side):
            side_map = actuator_by_base_and_side[base_name]
            if set(side_map.keys()) == {"l", "r"}:
                pairs.append((side_map["l"], side_map["r"]))
            else:
                incomplete_pairs[base_name] = sorted(side_map.keys())

        if incomplete_pairs:
            raise ValueError(
                "Could not construct complete left/right human muscle activation pairs. "
                f"Incomplete bases: {incomplete_pairs}"
            )
        if len(pairs) != 11:
            pair_names = [(left["name"], right["name"]) for left, right in pairs]
            raise ValueError(
                "Expected exactly 11 left/right human muscle activation pairs for the "
                f"22-muscle model, got {len(pairs)}: {pair_names}"
            )
        max_activation_index = max(
            max(left["activation_index"], right["activation_index"])
            for left, right in pairs
        )
        if max_activation_index >= self.sim.model.na:
            raise ValueError(
                "Muscle activation pair index exceeds MuJoCo activation state length: "
                f"max_activation_index={max_activation_index}, na={self.sim.model.na}"
            )
        return pairs

    def _reset_muscle_activation_symmetry_state(self):
        pair_count = len(getattr(self, "_muscle_activation_symmetry_pairs", []))
        self._muscle_activation_symmetry_left_integral = np.zeros(pair_count, dtype=float)
        self._muscle_activation_symmetry_right_integral = np.zeros(pair_count, dtype=float)
        self._muscle_activation_symmetry_cycle_duration = 0.0
        self._muscle_activation_symmetry_start_side = None
        self._muscle_activation_symmetry_last_event_side = None
        self._muscle_activation_symmetry_seen_opposite = False
        self._last_muscle_activation_symmetry_index = 0.0
        self._last_muscle_activation_symmetry_cycle_duration = 0.0

    @staticmethod
    def compute_muscle_activation_asymmetry_index(left_integral, right_integral, epsilon=1e-8):
        left_integral = np.asarray(left_integral, dtype=float)
        right_integral = np.asarray(right_integral, dtype=float)
        numerator = float(np.sum(np.abs(left_integral - right_integral)))
        denominator = float(np.sum(left_integral + right_integral))
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= epsilon:
            return 0.0
        asymmetry_index = numerator / (denominator + epsilon)
        return float(np.clip(asymmetry_index, 0.0, 1.0))

    @staticmethod
    def compute_muscle_activation_symmetry_penalty(left_integral, right_integral, cycle_duration, epsilon=1e-8):
        if not np.isfinite(cycle_duration) or cycle_duration <= 0.0:
            return 0.0, 0.0
        asymmetry_index = MyoAssistLegBase.compute_muscle_activation_asymmetry_index(
            left_integral,
            right_integral,
            epsilon=epsilon,
        )
        penalty = -float(cycle_duration) * asymmetry_index
        if not np.isfinite(penalty):
            return 0.0, 0.0
        return float(penalty), float(asymmetry_index)

    def _accumulate_muscle_activation_symmetry_integral(self, muscle_activations):
        for pair_index, (left_pair, right_pair) in enumerate(self._muscle_activation_symmetry_pairs):
            left_index = left_pair["activation_index"]
            right_index = right_pair["activation_index"]
            if left_index >= len(muscle_activations) or right_index >= len(muscle_activations):
                raise ValueError(
                    "Muscle activation vector is shorter than paired activation indices: "
                    f"len={len(muscle_activations)}, pair={left_pair['name']}/{right_pair['name']}, "
                    f"indices={left_index}/{right_index}"
                )
            self._muscle_activation_symmetry_left_integral[pair_index] += float(muscle_activations[left_index]) * self.dt
            self._muscle_activation_symmetry_right_integral[pair_index] += float(muscle_activations[right_index]) * self.dt
        self._muscle_activation_symmetry_cycle_duration += self.dt

    def _start_muscle_activation_symmetry_cycle(self, side):
        self._muscle_activation_symmetry_left_integral[:] = 0.0
        self._muscle_activation_symmetry_right_integral[:] = 0.0
        self._muscle_activation_symmetry_cycle_duration = 0.0
        self._muscle_activation_symmetry_start_side = side
        self._muscle_activation_symmetry_last_event_side = side
        self._muscle_activation_symmetry_seen_opposite = False

    def _update_muscle_activation_symmetry_penalty(self, muscle_activations, step_event_side):
        if self._muscle_activation_symmetry_start_side is not None:
            self._accumulate_muscle_activation_symmetry_integral(muscle_activations)

        if step_event_side not in ("left", "right"):
            return 0.0

        if self._muscle_activation_symmetry_start_side is None:
            self._start_muscle_activation_symmetry_cycle(step_event_side)
            return 0.0

        if step_event_side == self._muscle_activation_symmetry_last_event_side:
            self._start_muscle_activation_symmetry_cycle(step_event_side)
            return 0.0

        self._muscle_activation_symmetry_last_event_side = step_event_side
        if step_event_side != self._muscle_activation_symmetry_start_side:
            self._muscle_activation_symmetry_seen_opposite = True
            return 0.0

        if not self._muscle_activation_symmetry_seen_opposite:
            self._start_muscle_activation_symmetry_cycle(step_event_side)
            return 0.0

        cycle_duration = self._muscle_activation_symmetry_cycle_duration
        penalty, asymmetry_index = self.compute_muscle_activation_symmetry_penalty(
            self._muscle_activation_symmetry_left_integral,
            self._muscle_activation_symmetry_right_integral,
            cycle_duration,
        )
        self._last_muscle_activation_symmetry_index = asymmetry_index
        self._last_muscle_activation_symmetry_cycle_duration = cycle_duration
        self._start_muscle_activation_symmetry_cycle(step_event_side)
        return penalty
    def _get_max_joint_constraint_force(self):
        max_constraint_force = 0
        for sensor_name in self.joint_limit_sensor_keys:
            sensor_data = self.sim.data.sensor(sensor_name).data[0].copy()
            max_constraint_force = max(max_constraint_force, np.max(np.abs(sensor_data)))
        return max_constraint_force
    # ============ Custon Function ==============
    def _reset_properties_per_step(self):
        self._prev_pelvis_tx_pos = self.sim.data.body('pelvis').xpos[0]
        self._prev_step_time = self.sim.data.time
        self._activation_square_sum = 0
        self._footstep_delta_time = 0
        self._delta_velocity_sum = 0
    def _reset_reward_per_step(self):
        # prev reward save for non-sparse reward ( helpful for training? )
        self.reward_muscle_activation_penalty_per_step = 0
        self.reward_average_velocity_per_step = 0
        self.reward_footstep_delta_time = 0
        
    def _reset_heel_strike_buffer(self):
        self._r_heel_striking_value_buffer = []
        self._l_heel_striking_value_buffer = []
        self._last_heel_strike_foot = ""

    def _reset_gait_symmetry_reward_state(self):
        """Reset episode-local buffers used only by gait symmetry reward."""
        # Gait symmetry reward: timestamp of the most recent valid alternating heel strike.
        self._symmetry_last_heel_strike_time = None
        # Gait symmetry reward: side of the most recent valid alternating heel strike.
        self._symmetry_last_heel_strike_side = None
        # Gait symmetry reward: previous left-to-right or right-to-left step duration.
        self._symmetry_previous_step_duration = None
        # Gait symmetry reward: latest completed left stance duration.
        self._symmetry_latest_left_stance_duration = None
        # Gait symmetry reward: latest completed right stance duration.
        self._symmetry_latest_right_stance_duration = None
        # Gait symmetry reward: contact onset timestamp for each foot.
        self._symmetry_contact_start_time = {"left": None, "right": None}
        # Gait symmetry reward: independent previous contact state used only by the new reward.
        self._symmetry_previous_contact_state = {"left": False, "right": False}
        # Gait symmetry reward: independent contact confirmation buffers using the existing heel-strike rule.
        self._symmetry_contact_force_buffer = {"left": [], "right": []}

    def _update_gait_symmetry_contact_state(self):
        """Update independent contact and stance-duration state without modifying existing contact logic."""
        foot_force_by_side = {
            "left": self._get_foot_force("l"),
            "right": self._get_foot_force("r"),
        }
        for side, foot_force in foot_force_by_side.items():
            force_buffer = self._symmetry_contact_force_buffer[side]
            force_buffer.append(foot_force)
            if len(force_buffer) > 3:
                del force_buffer[:-3]
            contact_state = len(force_buffer) >= 3 and min(force_buffer[-3:]) > 0.1
            previous_contact_state = self._symmetry_previous_contact_state[side]

            if contact_state and not previous_contact_state:
                self._symmetry_contact_start_time[side] = self.sim.data.time
            elif not contact_state and previous_contact_state:
                contact_start_time = self._symmetry_contact_start_time[side]
                if contact_start_time is not None:
                    stance_duration = self.sim.data.time - contact_start_time
                    if stance_duration > 0:
                        if side == "left":
                            self._symmetry_latest_left_stance_duration = stance_duration
                        else:
                            self._symmetry_latest_right_stance_duration = stance_duration
                self._symmetry_contact_start_time[side] = None

            self._symmetry_previous_contact_state[side] = contact_state

    def _compute_step_symmetry_score(self, current_step_duration):
        """Compute normalized left-right alternating step-time symmetry score."""
        if self._symmetry_previous_step_duration is None:
            return 0.0, False
        eps = 1e-8
        step_asymmetry = abs(current_step_duration - self._symmetry_previous_step_duration) / (
            0.5 * (current_step_duration + self._symmetry_previous_step_duration) + eps
        )
        step_asymmetry = np.clip(step_asymmetry, 0.0, 1.0)
        step_symmetry_score = np.exp(-20.0 * step_asymmetry**2)
        return float(step_symmetry_score), True

    def _compute_stance_symmetry_score(self):
        """Compute normalized left-right stance-duration symmetry score."""
        if (
            self._symmetry_latest_left_stance_duration is None
            or self._symmetry_latest_right_stance_duration is None
        ):
            return 0.0, False
        eps = 1e-8
        stance_asymmetry = abs(
            self._symmetry_latest_left_stance_duration - self._symmetry_latest_right_stance_duration
        ) / (
            0.5 * (
                self._symmetry_latest_left_stance_duration
                + self._symmetry_latest_right_stance_duration
            ) + eps
        )
        stance_asymmetry = np.clip(stance_asymmetry, 0.0, 1.0)
        stance_symmetry_score = np.exp(-20.0 * stance_asymmetry**2)
        return float(stance_symmetry_score), True

    def _get_gait_symmetry_reward(self, heel_strike_detected):
        """Return an event-based gait symmetry score in [0, 1], or zero on non-event frames."""
        if not heel_strike_detected:
            return 0.0

        current_side = self._last_heel_strike_foot
        if current_side not in ("left", "right"):
            return 0.0

        current_time = self.sim.data.time
        if self._symmetry_last_heel_strike_side is None:
            self._symmetry_last_heel_strike_side = current_side
            self._symmetry_last_heel_strike_time = current_time
            return 0.0
        if current_side == self._symmetry_last_heel_strike_side:
            return 0.0

        current_step_duration = current_time - self._symmetry_last_heel_strike_time
        if current_step_duration <= 0:
            return 0.0

        step_symmetry_score, step_score_valid = self._compute_step_symmetry_score(current_step_duration)
        stance_symmetry_score, stance_score_valid = self._compute_stance_symmetry_score()

        valid_scores = []
        if step_score_valid:
            valid_scores.append(step_symmetry_score)
        if stance_score_valid:
            valid_scores.append(stance_symmetry_score)

        self._symmetry_previous_step_duration = current_step_duration
        self._symmetry_last_heel_strike_side = current_side
        self._symmetry_last_heel_strike_time = current_time

        if valid_scores:
            gait_symmetry_score = float(np.mean(valid_scores))
        else:
            gait_symmetry_score = 0.0
        return float(np.clip(gait_symmetry_score, 0.0, 1.0))

    def _reset_rearfoot_participation_state(self):
        self._rearfoot_stance_state = {
            "l": {
                "stance_active": False,
                "toe_seen": False,
                "rearfoot_seen": False,
                "contact_count": 0,
                "no_contact_count": 0,
                "rearfoot_contact_count": 0,
            },
            "r": {
                "stance_active": False,
                "toe_seen": False,
                "rearfoot_seen": False,
                "contact_count": 0,
                "no_contact_count": 0,
                "rearfoot_contact_count": 0,
            },
        }

    def _get_rearfoot_participation_contacts(self, foot_side_alphabet):
        toe_contact = self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0] > 0.0
        rearfoot_contact = self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0] > 0.0
        return bool(toe_contact), bool(rearfoot_contact)

    @staticmethod
    def _update_rearfoot_participation_stance_state(stance_state, toe_contact, rearfoot_contact):
        foot_contact = toe_contact or rearfoot_contact

        if foot_contact:
            stance_state["contact_count"] += 1
            stance_state["no_contact_count"] = 0
        else:
            stance_state["contact_count"] = 0

        if rearfoot_contact:
            stance_state["rearfoot_contact_count"] += 1
        else:
            stance_state["rearfoot_contact_count"] = 0

        if not stance_state["stance_active"] and stance_state["contact_count"] >= 2:
            stance_state["stance_active"] = True
            stance_state["toe_seen"] = False
            stance_state["rearfoot_seen"] = False

        if stance_state["stance_active"]:
            stance_state["toe_seen"] = stance_state["toe_seen"] or toe_contact
            stance_state["rearfoot_seen"] = (
                stance_state["rearfoot_seen"] or stance_state["rearfoot_contact_count"] >= 2
            )

        if stance_state["stance_active"] and not foot_contact:
            stance_state["no_contact_count"] += 1

            if stance_state["no_contact_count"] >= 3:
                toe_only_stance = stance_state["toe_seen"] and not stance_state["rearfoot_seen"]
                stance_state["stance_active"] = False
                stance_state["toe_seen"] = False
                stance_state["rearfoot_seen"] = False
                stance_state["contact_count"] = 0
                stance_state["no_contact_count"] = 0
                stance_state["rearfoot_contact_count"] = 0
                return -1.0 if toe_only_stance else 0.0

        return 0.0

    def _calculate_rearfoot_participation_penalty(self):
        if self.rwd_keys_wt.get("rearfoot_participation_penalty", 0.0) == 0.0:
            return 0.0
        if self._get_done():
            self._reset_rearfoot_participation_state()
            return 0.0

        penalty = 0.0
        for foot_side_alphabet in ("l", "r"):
            toe_contact, rearfoot_contact = self._get_rearfoot_participation_contacts(foot_side_alphabet)
            penalty += self._update_rearfoot_participation_stance_state(
                self._rearfoot_stance_state[foot_side_alphabet],
                toe_contact,
                rearfoot_contact,
            )
        return float(penalty)

    def _reset_pathological_foot_contact_state(self):
        self._pathological_region_contact_confirm_time = 0.02
        self._pathological_foot_contact_state = {
            "l": self._make_pathological_foot_state(),
            "r": self._make_pathological_foot_state(),
        }
        self._premature_heel_off_diagnostics = {
            "heel_off_ratio_l": None,
            "heel_off_ratio_r": None,
            "valid_heel_off_event_count_l": 0,
            "valid_heel_off_event_count_r": 0,
            "premature_heel_off_count_l": 0,
            "premature_heel_off_count_r": 0,
        }

    @staticmethod
    def _make_pathological_foot_state():
        return {
            "stance_active": False,
            "heel_contact": False,
            "toe_contact": False,
            "foot_contact": False,
            "prev_foot_contact": False,
            "heel_raw_contact": False,
            "toe_raw_contact": False,
            "heel_raw_start_time": None,
            "toe_raw_start_time": None,
            "heel_onset_time": None,
            "toe_onset_time": None,
            "stance_start_time": None,
            "heel_off_time": None,
            "heel_off_raw_start_time": None,
            "toe_first_settled": False,
            "rearfoot_persistence_settled": False,
            "waiting_opposite_toe_off": False,
            "heel_seen": False,
            "toe_seen": False,
        }

    def _get_pathological_region_contacts(self, foot_side_alphabet):
        heel_contact = self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0] > 0.0
        toe_contact = self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0] > 0.0
        return bool(heel_contact), bool(toe_contact)

    def _update_pathological_region_contact(
        self,
        foot_state,
        *,
        region_key,
        raw_contact,
        current_time,
    ):
        raw_key = f"{region_key}_raw_contact"
        raw_start_key = f"{region_key}_raw_start_time"
        contact_key = f"{region_key}_contact"
        onset_key = f"{region_key}_onset_time"

        if raw_contact:
            if not foot_state[raw_key]:
                foot_state[raw_start_key] = current_time
            foot_state[raw_key] = True

            raw_start_time = foot_state[raw_start_key]
            if raw_start_time is not None:
                raw_duration = current_time - raw_start_time + self.dt
                if raw_duration + 1e-12 >= self._pathological_region_contact_confirm_time:
                    foot_state[contact_key] = True
                    if foot_state[onset_key] is None:
                        foot_state[onset_key] = raw_start_time
        else:
            foot_state[raw_key] = False
            foot_state[raw_start_key] = None
            foot_state[contact_key] = False

    def _start_pathological_footfall(self, foot_side_alphabet):
        foot_state = self._pathological_foot_contact_state[foot_side_alphabet]
        opposite_side = "l" if foot_side_alphabet == "r" else "r"
        opposite_state = self._pathological_foot_contact_state[opposite_side]

        foot_state["stance_active"] = True
        onset_times = [
            time
            for time in (foot_state["heel_onset_time"], foot_state["toe_onset_time"])
            if time is not None
        ]
        foot_state["stance_start_time"] = min(onset_times) if onset_times else self.sim.data.time
        foot_state["heel_off_time"] = None
        foot_state["heel_off_raw_start_time"] = None
        foot_state["toe_first_settled"] = False
        foot_state["rearfoot_persistence_settled"] = False
        foot_state["waiting_opposite_toe_off"] = bool(opposite_state["foot_contact"])
        foot_state["heel_seen"] = bool(foot_state["heel_contact"])
        foot_state["toe_seen"] = bool(foot_state["toe_contact"])

    def _end_pathological_footfall(self, foot_state):
        foot_state["stance_active"] = False
        foot_state["heel_onset_time"] = None
        foot_state["toe_onset_time"] = None
        foot_state["stance_start_time"] = None
        foot_state["heel_off_time"] = None
        foot_state["heel_off_raw_start_time"] = None
        foot_state["toe_first_settled"] = False
        foot_state["rearfoot_persistence_settled"] = False
        foot_state["waiting_opposite_toe_off"] = False
        foot_state["heel_seen"] = False
        foot_state["toe_seen"] = False

    def _update_pathological_toe_first_penalty(self, foot_state, current_time):
        if foot_state["toe_first_settled"]:
            return 0.0

        heel_onset_time = foot_state["heel_onset_time"]
        toe_onset_time = foot_state["toe_onset_time"]
        tolerance = self._pathological_region_contact_confirm_time

        if toe_onset_time is not None and heel_onset_time is not None:
            foot_state["toe_first_settled"] = True
            return -1.0 if toe_onset_time + tolerance < heel_onset_time else 0.0

        if toe_onset_time is not None and current_time - toe_onset_time > tolerance:
            foot_state["toe_first_settled"] = True
            return -1.0

        if heel_onset_time is not None and current_time - heel_onset_time > tolerance:
            foot_state["toe_first_settled"] = True
            return 0.0

        return 0.0

    def _update_pathological_persistence_penalty(self, foot_side_alphabet):
        foot_state = self._pathological_foot_contact_state[foot_side_alphabet]
        if (
            foot_state["rearfoot_persistence_settled"]
            or not foot_state["waiting_opposite_toe_off"]
        ):
            return 0.0

        opposite_side = "l" if foot_side_alphabet == "r" else "r"
        opposite_state = self._pathological_foot_contact_state[opposite_side]
        opposite_toe_off = opposite_state["prev_foot_contact"] and not opposite_state["foot_contact"]
        if not opposite_toe_off:
            return 0.0

        foot_state["rearfoot_persistence_settled"] = True
        foot_state["waiting_opposite_toe_off"] = False
        return 0.0 if foot_state["heel_contact"] else -1.0

    def _update_premature_heel_off_event(self, foot_state, current_time):
        if foot_state["heel_contact"]:
            foot_state["heel_seen"] = True
        if foot_state["toe_contact"]:
            foot_state["toe_seen"] = True

        if (
            foot_state["heel_off_time"] is not None
            or not foot_state["heel_seen"]
            or not foot_state["toe_seen"]
        ):
            return

        if foot_state["toe_contact"] and not foot_state["heel_contact"]:
            if foot_state["heel_off_raw_start_time"] is None:
                foot_state["heel_off_raw_start_time"] = current_time
            raw_duration = current_time - foot_state["heel_off_raw_start_time"] + self.dt
            if raw_duration + 1e-12 >= self._pathological_region_contact_confirm_time:
                foot_state["heel_off_time"] = foot_state["heel_off_raw_start_time"]
        else:
            foot_state["heel_off_raw_start_time"] = None

    def _settle_premature_heel_off_penalty(self, foot_side_alphabet, foot_state, stance_end_time):
        stance_start_time = foot_state["stance_start_time"]
        heel_off_time = foot_state["heel_off_time"]
        if (
            stance_start_time is None
            or heel_off_time is None
            or not foot_state["heel_seen"]
            or not foot_state["toe_seen"]
        ):
            return 0.0

        stance_duration = stance_end_time - stance_start_time
        if stance_duration <= 0.0:
            return 0.0

        heel_off_ratio = (heel_off_time - stance_start_time) / stance_duration
        ratio_key = f"heel_off_ratio_{foot_side_alphabet}"
        valid_count_key = f"valid_heel_off_event_count_{foot_side_alphabet}"
        premature_count_key = f"premature_heel_off_count_{foot_side_alphabet}"
        self._premature_heel_off_diagnostics[ratio_key] = float(heel_off_ratio)
        self._premature_heel_off_diagnostics[valid_count_key] += 1

        min_ratio = self.MIN_HEEL_OFF_STANCE_RATIO
        if heel_off_ratio >= min_ratio:
            return 0.0

        self._premature_heel_off_diagnostics[premature_count_key] += 1
        return -float(np.clip((min_ratio - heel_off_ratio) / min_ratio, 0.0, 1.0))

    def _calculate_pathological_foot_contact_penalties(self):
        if (
            self.rwd_keys_wt.get("toe_first_penalty", 0.0) == 0.0
            and self.rwd_keys_wt.get("rearfoot_persistence_penalty", 0.0) == 0.0
            and self.rwd_keys_wt.get("premature_heel_off_penalty", 0.0) == 0.0
        ):
            return 0.0, 0.0, 0.0
        if self._get_done():
            self._reset_pathological_foot_contact_state()
            return 0.0, 0.0, 0.0

        current_time = self.sim.data.time
        for foot_side_alphabet in ("l", "r"):
            foot_state = self._pathological_foot_contact_state[foot_side_alphabet]
            heel_contact, toe_contact = self._get_pathological_region_contacts(foot_side_alphabet)
            self._update_pathological_region_contact(
                foot_state,
                region_key="heel",
                raw_contact=heel_contact,
                current_time=current_time,
            )
            self._update_pathological_region_contact(
                foot_state,
                region_key="toe",
                raw_contact=toe_contact,
                current_time=current_time,
            )
            foot_state["prev_foot_contact"] = foot_state["foot_contact"]
            foot_state["foot_contact"] = foot_state["heel_contact"] or foot_state["toe_contact"]

        toe_first_penalty = 0.0
        rearfoot_persistence_penalty = 0.0
        premature_heel_off_penalty = 0.0
        for foot_side_alphabet in ("l", "r"):
            foot_state = self._pathological_foot_contact_state[foot_side_alphabet]

            if foot_state["foot_contact"] and not foot_state["prev_foot_contact"]:
                self._start_pathological_footfall(foot_side_alphabet)

            if foot_state["stance_active"]:
                toe_first_penalty += self._update_pathological_toe_first_penalty(
                    foot_state,
                    current_time,
                )
                rearfoot_persistence_penalty += self._update_pathological_persistence_penalty(
                    foot_side_alphabet
                )
                self._update_premature_heel_off_event(foot_state, current_time)

            if foot_state["stance_active"] and not foot_state["foot_contact"]:
                premature_heel_off_penalty += self._settle_premature_heel_off_penalty(
                    foot_side_alphabet,
                    foot_state,
                    current_time,
                )
                self._end_pathological_footfall(foot_state)

        return (
            float(toe_first_penalty),
            float(rearfoot_persistence_penalty),
            float(premature_heel_off_penalty),
        )

    def _detect_heel_strike(self):
        r_foot_force = self._get_foot_force("r")
        l_foot_force = self._get_foot_force("l")

        self._r_heel_striking_value_buffer.append( r_foot_force)
        self._l_heel_striking_value_buffer.append(l_foot_force)

        if len(self._r_heel_striking_value_buffer) > 2:
            last_three_min = min(self._r_heel_striking_value_buffer[-3:])
            if last_three_min > 0.1 and self._last_heel_strike_foot != "right":
                self._last_heel_strike_foot = "right"
                # print("DEBUG:: right heel strike")
                return True
        if len(self._l_heel_striking_value_buffer) > 2:
            last_three_min = min(self._l_heel_striking_value_buffer[-3:])
            if last_three_min > 0.1 and self._last_heel_strike_foot != "left":
                self._last_heel_strike_foot = "left"
                # print("DEBUG:: left heel strike")
                return True
        return False
    def _get_foot_force(self, foot_side_alphabet:str):
        foot_force = self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0] + self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0]
        return foot_force

    # To override
    def _initialize_pose(self):
        self.sim.data.qpos[:] = self.sim.model.key_qpos[0][:]
        self.just_forward()
