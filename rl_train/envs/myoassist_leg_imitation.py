import collections
import numpy as np
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
################################################################


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 original_reward_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params:ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 evaluate_timeout_seconds: float | None = None,
                 verbose=1):
        super().__init__(log_rollout_freq=log_rollout_freq, 
                         evaluate_freq=evaluate_freq,
                         log_handler=log_handler,
                         evaluate_timeout_seconds=evaluate_timeout_seconds,
                         verbose=verbose)
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params
        

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0
    #called after all envs step done
    def _on_step(self) -> bool:
        # print("======================self.locals    ======================")
        # # pprint.pprint(self.locals)
        # print(f"DEBUG:: {len(self.locals['infos'])=}")
        # for idx, info in enumerate(self.locals['infos']):
        #     print(f"DEBUG:: {idx=} {info['rwd_dict']=}")
        # print("======================self.locals    ======================")

        subprocvec_env:SubprocVecEnv = self.model.get_env()
        # print(f"DEBUG:: {subprocvec_env=}")
        # print(f"DEBUG:: {subprocvec_env.env_method('subproc_env_test', 'This is param from learning callback')=}")
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]
        

        super()._on_step()
            
        return True
    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True) -> "ImitationTrainCheckpointData":
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()
        
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        

        ## ARA (Disabled)

        # print(f"DEBUG:: {self.reward_accumulate=}")
        # joint_rewards = {}
        # for key in self.reward_accumulate.keys():
        #     # print(f"DEBUG:: {key=} {self.reward_accumulate[key]=}")
        #     if MyoLeg18Imitation.Q_POS_DIFF_REWARD_KEY_PREFIX in key:
        #         joint_rewards[key] = self.reward_accumulate[key]
        #     self.reward_accumulate[key] = 0
        # reward_mean = 0
        # for key in joint_rewards.keys():
        #     reward_mean += joint_rewards[key]
        # reward_mean /= len(joint_rewards)
        # joint_reward_deviations = {key: (joint_rewards[key] - reward_mean)/reward_mean for key in joint_rewards.keys()}

        # for key in joint_reward_deviations.keys():
        #     new_reward_weight = getattr(self._reward_weights, key) - self._auto_reward_adjust_params.learning_rate * joint_reward_deviations[key]
        #     setattr(self._reward_weights, key, new_reward_weight)
        # subprocvec_env:SubprocVecEnv = self.model.get_env()
        # subprocvec_env.env_method('set_reward_weights', self._reward_weights)
        # print(f"DEBUG:: {self._reward_weights=}")


##############################################################################



class MyoAssistLegImitation(MyoAssistLegBase):
    REFERENCE_MIRROR_JOINT_PAIRS = (
        ("hip_flexion_r", "hip_flexion_l"),
        ("knee_angle_r", "knee_angle_l"),
        ("ankle_angle_r", "ankle_angle_l"),
    )
    REFERENCE_MIRROR_JOINT_SIGNS = {
        "hip_flexion_r": 1.0,
        "hip_flexion_l": 1.0,
        "knee_angle_r": 1.0,
        "knee_angle_l": 1.0,
        "ankle_angle_r": 1.0,
        "ankle_angle_l": 1.0,
    }
    
    # automatically inherit from MyoAssistLegBase
    # DEFAULT_OBS_KEYS = ['qpos',
    #                     'qvel',
    #                     'act',
    #                     'target_velocity',
    #                     ]

    def _setup(self,*,
            env_params:ImitationTrainSessionConfig.EnvParams,
            reference_data:dict|None = None,
            loop_reference_data:bool = False,
            **kwargs,
        ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._flag_mirror_reference = bool(getattr(env_params, "flag_mirror_reference", False))
        self._mirror_reference_probability = float(getattr(env_params, "mirror_reference_probability", 0.5))
        if not 0.0 <= self._mirror_reference_probability <= 1.0:
            raise ValueError("mirror_reference_probability must be in [0, 1]")
        self._mirror_reference_episode = False
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)
        self._setup_reference_mirror_mapping()

        super()._setup(env_params=env_params,
                       **kwargs,
                       )

        
        
    def set_reward_weights(self, reward_keys_and_weights:TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights
    # override from MujocoEnv
    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qpos[0].copy() - self._get_reference_qpos(key, self._imitation_index)
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict
    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = self._get_reference_speed_ratio(self._imitation_index)

        def get_qvel_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qvel[0].copy() - self._get_reference_qvel(key, self._imitation_index) * speed_ratio_to_target_velocity
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict
    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])
    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])
    
    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))
        
        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum([q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()])
        qvel_imitation_rewards = np.sum([dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()])

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update({
            'qpos_imitation_rewards': qpos_imitation_rewards,
            'qvel_imitation_rewards': qvel_imitation_rewards,
            'end_effector_imitation_reward': anchor_reward
        })

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info
    

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),
        })
        # Calculate final reward
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)
        
        return rwd_dict
    

    def _follow_reference_motion(self, is_x_follow:bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._get_reference_qpos(key, self._imitation_index)
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = self._get_reference_speed_ratio(self._imitation_index)
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = self._get_reference_qvel(key, self._imitation_index) * speed_ratio_to_target_velocity
    def imitation_step(self, is_x_follow:bool, specific_index:int|None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass
    
    # override
    def step(self, a, **kwargs):
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True
        
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if isinstance(info, dict):
            info["mirror_reference"] = bool(self._mirror_reference_episode)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray:np.ndarray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(np.abs(q_diff_nparray) >self._out_of_trajectory_threshold)
            terminated = terminated or is_out_of_trajectory
        
        return (next_obs, reward, terminated, truncated, info)
        
    
    def setup_reference_data(self, data:dict|None):
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            # self._follow_reference_motion(False)
            self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]
        else:
            raise ValueError("Reference data is not set")

    def _setup_reference_mirror_mapping(self) -> None:
        series_data = self._reference_data["series_data"]
        self._reference_mirror_source_by_key = {}
        for right_name, left_name in self.REFERENCE_MIRROR_JOINT_PAIRS:
            for prefix in ("q", "dq"):
                right_key = f"{prefix}_{right_name}"
                left_key = f"{prefix}_{left_name}"
                if right_key not in series_data or left_key not in series_data:
                    raise ValueError(
                        "reference mirror pair missing from reference_data: "
                        f"{right_key}, {left_key}"
                    )
            self._reference_mirror_source_by_key[right_name] = left_name
            self._reference_mirror_source_by_key[left_name] = right_name

    def _sample_mirror_reference_episode(self) -> bool:
        if not self._flag_mirror_reference:
            return False
        return bool(self.np_random.random() < self._mirror_reference_probability)

    def _get_reference_source_key_and_sign(self, joint_name: str) -> tuple[str, float]:
        if not self._mirror_reference_episode:
            return joint_name, 1.0
        source_name = self._reference_mirror_source_by_key.get(joint_name, joint_name)
        sign = self.REFERENCE_MIRROR_JOINT_SIGNS.get(joint_name, 1.0)
        return source_name, sign

    def _get_reference_qpos(self, joint_name: str, reference_index: int) -> float:
        if reference_index is None:
            reference_index = 0
        source_name, sign = self._get_reference_source_key_and_sign(joint_name)
        return float(sign * self._reference_data["series_data"][f"q_{source_name}"][reference_index])

    def _get_reference_qvel(self, joint_name: str, reference_index: int) -> float:
        if reference_index is None:
            reference_index = 0
        source_name, sign = self._get_reference_source_key_and_sign(joint_name)
        return float(sign * self._reference_data["series_data"][f"dq_{source_name}"][reference_index])

    def _get_reference_speed_ratio(self, reference_index: int) -> float:
        if reference_index is None:
            reference_index = 0
        ref_pelvis_velocity = float(self._reference_data["series_data"]["dq_pelvis_tx"][reference_index])
        return float(self._target_velocity / ref_pelvis_velocity)

    def get_reference_qpos_map(self, reference_index: int, *, mirrored: bool | None = None) -> dict[str, float]:
        previous = self._mirror_reference_episode
        if mirrored is not None:
            self._mirror_reference_episode = bool(mirrored)
        try:
            return {
                key: self._get_reference_qpos(key, reference_index)
                for key in self.reference_data_keys
            }
        finally:
            self._mirror_reference_episode = previous

    def get_reference_qvel_map(self, reference_index: int, *, mirrored: bool | None = None) -> dict[str, float]:
        previous = self._mirror_reference_episode
        if mirrored is not None:
            self._mirror_reference_episode = bool(mirrored)
        try:
            return {
                key: self._get_reference_qvel(key, reference_index)
                for key in self.reference_data_keys
            }
        finally:
            self._mirror_reference_episode = previous

    def reset(self, **kwargs):
        rng = np.random.default_rng()# TODO: refactoring random to use seed
        
        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(0, int(self._reference_data_length * 0.8))
        else:
            self._imitation_index = 0
        self._mirror_reference_episode = self._sample_mirror_reference_episode()
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(False)
        
        obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        return obs

    # override
    def _initialize_pose(self):
        super()._initialize_pose()
