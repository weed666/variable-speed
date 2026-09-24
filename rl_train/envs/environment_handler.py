import numpy as np
import json
import torch
import torch._dynamo  # Preload before MyoSuite/MuJoCo extensions to avoid lazy optimizer import crashes.
from stable_baselines3.common.vec_env import SubprocVecEnv
from myosuite.utils import gym
from rl_train.utils.data_types import DictionableDataclass
import os
import multiprocessing
import math
from rl_train.train.train_configs.config import TrainSessionConfigBase
class EnvironmentHandler:
    @staticmethod
    def restore_sb3_save_params(model):
        if "_get_torch_save_params" in getattr(model, "__dict__", {}):
            delattr(model, "_get_torch_save_params")
        return model

    @staticmethod
    def select_subproc_start_method() -> str:
        methods = multiprocessing.get_all_start_methods()
        if "forkserver" in methods:
            return "forkserver"
        if "spawn" in methods:
            return "spawn"
        raise RuntimeError(
            "Neither forkserver nor spawn is available; "
            "unsafe fork fallback is not allowed."
        )

    @staticmethod
    def create_environment(config, is_rendering_on:bool, is_evaluate_mode:bool = False):

        ref_data_dict = EnvironmentHandler.load_reference_data(config)
    
        # Base gym.make arguments
        gym_make_args = {
            'seed': config.env_params.seed,
            'model_path': config.env_params.model_path,
            'env_params': config.env_params,
            'is_evaluate_mode': is_evaluate_mode
        }
        
        # Add reference_data only if it exists
        if ref_data_dict is not None:
            gym_make_args['reference_data'] = ref_data_dict
        
        try:
            if is_rendering_on or config.env_params.num_envs == 1:
                print(f"{config.env_params.env_id=}")
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                if is_rendering_on:
                    env.mujoco_render_frames = True
                config.env_params.num_envs = 1
                config.ppo_params.n_steps = config.ppo_params.batch_size
            else:
                start_method = EnvironmentHandler.select_subproc_start_method()
                print(f"SubprocVecEnv start_method: {start_method}")
                env = SubprocVecEnv([lambda: (gym.make(config.env_params.env_id, 
                                                    **gym_make_args)).unwrapped 
                                for _ in range(config.env_params.num_envs)],
                                start_method=start_method)
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e
        return env

    @staticmethod
    def load_reference_data(config):
        # Check if config has reference_data_path attribute
        print("===================================================================")
        if not hasattr(config.env_params, 'reference_data_path'):
            print("No reference data path provided.")
            print("===================================================================")
            return None
            
        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("===================================================================")
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("===================================================================")
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(config.env_params.reference_data_path, allow_pickle=True)
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    def get_config_type_from_session_id(session_id):
        # from rl_train.envs import myo_leg_18_reward_per_step
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_stage3 import Stage3TrainSessionConfig
        from rl_train.train.train_configs.config_stage3_exo import Stage3ExoTeacherTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        # Create appropriate config based on env_id
        print(f"session_id: {session_id}")
        if session_id in ['myoAssistLeg-v0']:
            return TrainSessionConfigBase
        elif session_id == 'myoAssistLegAcceleration-v0':
            return Stage3TrainSessionConfig
        elif session_id == 'myoAssistLegAccelerationExo-v0':
            return Stage3ExoTeacherTrainSessionConfig
        elif session_id in ['myoAssistLegImitation-v0']:
            return ImitationTrainSessionConfig
        elif session_id == 'myoAssistLegImitationExo-v0':
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")
        

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        config_file_path = config_path
        with open(config_file_path, 'r') as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        
        from rl_train.envs import myoassist_leg_imitation
        from rl_train.utils import learning_callback
        if isinstance(config, ImitationTrainSessionConfig):
            custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                evaluate_timeout_seconds=config.logger_params.evaluate_timeout_seconds,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
            )
        else:
            custom_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                evaluate_timeout_seconds=config.logger_params.evaluate_timeout_seconds,
            )

        return custom_callback
    @staticmethod
    def get_stable_baselines3_model(config:TrainSessionConfigBase, env, trained_model_path:str|None=None):
        import stable_baselines3
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
        if config.env_params.env_id in ["myoAssistLegImitationExo-v0", "myoAssistLegAccelerationExo-v0"]:
            policy_class = HumanExoActorCriticPolicy
            print(f"Using HumanExoActorCriticPolicy")
        else:
            policy_class = HumanActorCriticPolicy
            print(f"Using HumanActorCriticPolicy")
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = stable_baselines3.PPO.load(trained_model_path,
                                            env=env,
                                            custom_objects = {"policy_class": policy_class},
                                            )
            EnvironmentHandler.restore_sb3_save_params(model)
        elif config.env_params.prev_trained_policy_path:
            print(f"Loading previous trained policy from {config.env_params.prev_trained_policy_path}")
            EnvironmentHandler.warn_stage5_lpf_mismatch_if_available(config)
            # when should I reset the (value)network?
            model = stable_baselines3.PPO.load(config.env_params.prev_trained_policy_path,
                                            env=env,
                                            custom_objects = {"policy_class": policy_class},

                                            # policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                                            verbose=2,
                                            **DictionableDataclass.to_dict(config.ppo_params),
                                            )
            EnvironmentHandler.restore_sb3_save_params(model)
            # print(f"Resetting network: {config.custom_policy_params.reset_shared_net_after_load=}, {config.custom_policy_params.reset_policy_net_after_load=}, {config.custom_policy_params.reset_value_net_after_load=}")
            model.policy.reset_network(reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                                    reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                                    reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load)
        else:
            model = stable_baselines3.PPO(
                policy=policy_class,
                env=env,
                policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
        EnvironmentHandler.apply_exo_teacher_freeze_if_requested(config, model)
        return model

    @staticmethod
    def _env_lpf_params(env_params) -> dict:
        enabled = bool(getattr(env_params, "exo_output_lpf_enabled", False))
        tau_s = getattr(env_params, "exo_output_lpf_tau_s", None)
        old_alpha = getattr(env_params, "exo_output_lpf_alpha", None)
        tau_s = None if tau_s is None or float(tau_s) <= 0.0 else float(tau_s)
        if tau_s is None and enabled and old_alpha is not None:
            raise ValueError(
                "exo_output_lpf_alpha is deprecated and cannot configure an enabled LPF. "
                "Use exo_output_lpf_tau_s."
            )
        if tau_s is None and enabled:
            raise ValueError("exo_output_lpf_tau_s is required when exo_output_lpf_enabled is true")
        tau_s = 0.1 if tau_s is None else float(tau_s)
        control_hz = float(getattr(env_params, "control_framerate"))
        dt_s = 1.0 / control_hz
        return {
            "exo_output_lpf_enabled": enabled,
            "exo_output_lpf_tau_s": tau_s,
            "exo_output_lpf_dt_s": dt_s,
            "exo_output_lpf_alpha_effective": 1.0 - math.exp(-dt_s / tau_s),
        }

    @staticmethod
    def _lpf_params_match(left: dict, right: dict) -> bool:
        left_enabled = bool(left.get("exo_output_lpf_enabled", False))
        right_enabled = bool(right.get("exo_output_lpf_enabled", False))
        if left_enabled != right_enabled:
            return False
        if not left_enabled and not right_enabled:
            return True
        return (
            abs(float(left.get("exo_output_lpf_tau_s", 0.1)) - float(right.get("exo_output_lpf_tau_s", 0.1)))
            <= 1e-12
            and abs(float(left.get("exo_output_lpf_dt_s", 0.0)) - float(right.get("exo_output_lpf_dt_s", 0.0)))
            <= 1e-12
        )

    @staticmethod
    def warn_stage5_lpf_mismatch_if_available(config:TrainSessionConfigBase) -> None:
        if getattr(config.env_params, "env_id", "") != "myoAssistLegAccelerationExo-v0":
            return
        if getattr(config.env_params, "training_stage", "") != "stage5_coadapt":
            return
        checkpoint_path = getattr(config.env_params, "prev_trained_policy_path", "")
        if not checkpoint_path:
            return
        checkpoint_abs = os.path.abspath(checkpoint_path)
        report_dir = os.path.dirname(checkpoint_abs)
        if not os.path.isdir(report_dir):
            return
        current_lpf = EnvironmentHandler._env_lpf_params(config.env_params)
        for file_name in sorted(os.listdir(report_dir)):
            if not file_name.endswith(".json"):
                continue
            report_path = os.path.join(report_dir, file_name)
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                continue
            output_checkpoint = report.get("output_checkpoint")
            if not output_checkpoint:
                continue
            if os.path.abspath(output_checkpoint) != checkpoint_abs:
                continue
            source_lpf = report.get("source_lpf_params")
            if not isinstance(source_lpf, dict):
                return
            if not EnvironmentHandler._lpf_params_match(source_lpf, current_lpf):
                raise ValueError(
                    "Stage5 LPF parameters differ from the Stage4 source "
                    f"used to create {checkpoint_abs}: source={source_lpf}, stage5={current_lpf}"
                )
            return
    @staticmethod
    def apply_exo_teacher_freeze_if_requested(config:TrainSessionConfigBase, model):
        if config.env_params.env_id != "myoAssistLegAccelerationExo-v0":
            return
        custom_policy_params = config.policy_params.custom_policy_params
        if not getattr(custom_policy_params, "freeze_human_actor", False):
            return
        model.policy.apply_human_freeze(
            freeze_log_std=getattr(custom_policy_params, "freeze_human_log_std", True),
            human_log_std_freeze_end=getattr(custom_policy_params, "human_log_std_freeze_end", 22),
        )
        EnvironmentHandler.sanitize_human_log_std_optimizer_state(
            config,
            model,
            human_log_std_freeze_end=getattr(custom_policy_params, "human_log_std_freeze_end", 22),
        )

        # EXO_TEACHER:
        # Do not rebuild the optimizer after same-structure E1 PPO.load().
        # SB3 restores policy.optimizer from the checkpoint and rebuilding it
        # would discard Exo/Critic Adam state. Human freezing is handled
        # separately by apply_human_freeze().

    @staticmethod
    def sanitize_human_log_std_optimizer_state(
        config:TrainSessionConfigBase,
        model,
        *,
        human_log_std_freeze_end:int = 22,
    ):
        if config.env_params.env_id != "myoAssistLegAccelerationExo-v0":
            return
        custom_policy_params = config.policy_params.custom_policy_params
        if not getattr(custom_policy_params, "freeze_human_actor", False):
            return
        if not getattr(custom_policy_params, "freeze_human_log_std", True):
            return

        log_std = getattr(model.policy, "log_std", None)
        optimizer = getattr(model.policy, "optimizer", None)
        if log_std is None or optimizer is None:
            return

        state = optimizer.state.get(log_std)
        if not state:
            return

        freeze_end = int(human_log_std_freeze_end)
        for state_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            state_tensor = state.get(state_key)
            if not isinstance(state_tensor, torch.Tensor):
                continue
            if state_tensor.shape != log_std.shape:
                continue
            state_tensor[:freeze_end] = 0

    @staticmethod
    def rebuild_policy_optimizer(config:TrainSessionConfigBase, model):
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(),
            lr=config.ppo_params.learning_rate,
            **model.policy.optimizer_kwargs,
        )
    @staticmethod
    def updateconfig_from_model_policy(config, model):
        pass
        # config.policy_info.extractor_policy_net = f"{model.policy.mlp_extractor.policy_net}"
        # config.policy_info.extractor_value_net = f"{model.policy.mlp_extractor.value_net}"
        # config.policy_info.action_net = f"{model.policy.action_net}"
        # config.policy_info.value_net = f"{model.policy.value_net}"
        # config.policy_info.ortho_init = f"{model.policy.ortho_init}"
        # config.policy_info.share_features_extractor = f"{model.policy.share_features_extractor}"
