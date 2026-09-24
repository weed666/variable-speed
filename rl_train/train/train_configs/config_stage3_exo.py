from dataclasses import dataclass, field

from rl_train.train.train_configs.config_stage3 import Stage3TrainSessionConfig


@dataclass
class Stage3ExoTeacherTrainSessionConfig(Stage3TrainSessionConfig):
    @dataclass
    class EnvParams(Stage3TrainSessionConfig.EnvParams):
        hip_muscle_activation_reward_weight: float = 2.0
        exo_power_reward_weight: float = 4.0
        exo_power_reward_alpha: float = 0.5
        exo_power_reward_mode: str = "positive"
        training_stage: str = "stage3_exo_teacher"
        teacher_regime: str = "mixed"
        task_pool_mode: str = "speed_level_pairs"
        delta_velocity_levels: list[float] = field(
            default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )
        speed_min: float = 0.9
        speed_max: float = 1.6
        dynamic_episode_duration: bool = False
        pre_hold_min_s: float = 1.0
        pre_hold_max_s: float = 2.0
        post_hold_min_s: float = 1.0
        post_hold_max_s: float = 2.0
        steady_speed_sampling: str = "uniform"
        steady_episode_duration_min_s: float = 4.0
        steady_episode_duration_max_s: float = 6.0
        exo_torque_limit_nm: float = 6.0
        torque_limit_reference_nm: float = 25.0
        exo_output_lpf_enabled: bool = False
        exo_output_lpf_tau_s: float = -1.0
        exo_output_lpf_alpha: float = None
        stage5_observation_extension: bool = False
        stage5_smat_power_magnitude_saturation_reward_weight: float = 0.0
        stage5_smat_torque_rate_penalty_weight: float = 0.0
        stage5_smat_power_alpha: float = 0.3
        stage5_smat_magnitude_beta: float = 0.15
        stage5_smat_saturation_lambda: float = 2.0
        stage5_smat_saturation_delta: float = 0.8
        stage5_smat_omega_scale_rad_s: float = 2.0

    env_params: EnvParams = field(default_factory=EnvParams)

    @dataclass
    class PolicyParams(Stage3TrainSessionConfig.PolicyParams):
        @dataclass
        class CustomPolicyParams(Stage3TrainSessionConfig.PolicyParams.CustomPolicyParams):
            freeze_human_actor: bool = True
            freeze_human_log_std: bool = True
            human_log_std_freeze_end: int = 22

        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

    policy_params: PolicyParams = field(default_factory=PolicyParams)
