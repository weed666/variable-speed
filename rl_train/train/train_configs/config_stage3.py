from dataclasses import dataclass, field

from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig


@dataclass
class Stage3TrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class RewardWeights(ImitationTrainSessionConfig.EnvParams.RewardWeights):
            qpos_imitation_rewards: dict = field(default_factory=dict)
            qvel_imitation_rewards: dict = field(default_factory=dict)
            end_effector_imitation_reward: float = 0.0

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

    env_params: EnvParams = field(default_factory=EnvParams)
