from __future__ import annotations

import copy
import unittest

import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler


CONFIG_PATH = "rl_train/train/train_configs/stage1_reference_mirror.json"
JOINT_PAIRS = (
    ("hip_flexion_r", "hip_flexion_l"),
    ("knee_angle_r", "knee_angle_l"),
    ("ankle_angle_r", "ankle_angle_l"),
)
PELVIS_KEYS = ("pelvis_tx", "pelvis_ty", "pelvis_tilt")


def load_test_config():
    config_type = EnvironmentHandler.get_config_type_from_session_id("myoAssistLegImitation-v0")
    cfg = EnvironmentHandler.get_session_config_from_path(CONFIG_PATH, config_type)
    cfg.env_params.num_envs = 1
    cfg.env_params.flag_random_ref_index = False
    cfg.env_params.prev_trained_policy_path = None
    cfg.ppo_params.device = "cpu"
    cfg.ppo_params.n_steps = 8
    cfg.ppo_params.batch_size = 8
    cfg.ppo_params.n_epochs = 1
    return cfg


class Stage1ReferenceMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_test_config()
        cls.env = EnvironmentHandler.create_environment(
            cls.config,
            is_rendering_on=False,
            is_evaluate_mode=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_mirror_involution_qpos_qvel(self):
        for idx in (0, 17, 123, 499):
            qpos_original = self.env.get_reference_qpos_map(idx, mirrored=False)
            qvel_original = self.env.get_reference_qvel_map(idx, mirrored=False)
            qpos_mirrored = self.env.get_reference_qpos_map(idx, mirrored=True)
            qvel_mirrored = self.env.get_reference_qvel_map(idx, mirrored=True)

            for right, left in JOINT_PAIRS:
                self.assertAlmostEqual(qpos_mirrored[right], qpos_original[left])
                self.assertAlmostEqual(qpos_mirrored[left], qpos_original[right])
                self.assertAlmostEqual(qvel_mirrored[right], qvel_original[left])
                self.assertAlmostEqual(qvel_mirrored[left], qvel_original[right])

                qpos_twice = copy.deepcopy(qpos_mirrored)
                qvel_twice = copy.deepcopy(qvel_mirrored)
                qpos_twice[right], qpos_twice[left] = qpos_twice[left], qpos_twice[right]
                qvel_twice[right], qvel_twice[left] = qvel_twice[left], qvel_twice[right]
                self.assertAlmostEqual(qpos_twice[right], qpos_original[right])
                self.assertAlmostEqual(qpos_twice[left], qpos_original[left])
                self.assertAlmostEqual(qvel_twice[right], qvel_original[right])
                self.assertAlmostEqual(qvel_twice[left], qvel_original[left])

    def test_pelvis_unchanged(self):
        for idx in (0, 17, 123, 499):
            qpos_original = self.env.get_reference_qpos_map(idx, mirrored=False)
            qvel_original = self.env.get_reference_qvel_map(idx, mirrored=False)
            qpos_mirrored = self.env.get_reference_qpos_map(idx, mirrored=True)
            qvel_mirrored = self.env.get_reference_qvel_map(idx, mirrored=True)
            for key in PELVIS_KEYS:
                self.assertEqual(qpos_mirrored[key], qpos_original[key])
                self.assertEqual(qvel_mirrored[key], qvel_original[key])

    def test_probability_extremes(self):
        self.env._flag_mirror_reference = True
        self.env._mirror_reference_probability = 0.0
        self.assertFalse(any(self.env._sample_mirror_reference_episode() for _ in range(20)))
        self.env._mirror_reference_probability = 1.0
        self.assertTrue(all(self.env._sample_mirror_reference_episode() for _ in range(20)))

    def test_current_observation_unchanged_by_flag(self):
        self.env.reset()
        obs_dict = self.env.get_obs_dict(self.env.sim)
        self.env._mirror_reference_episode = not self.env._mirror_reference_episode
        mirrored_obs_dict = self.env.get_obs_dict(self.env.sim)
        for key in ("qpos", "qvel", "act", "sensor"):
            np.testing.assert_array_equal(obs_dict[key], mirrored_obs_dict[key])


if __name__ == "__main__":
    unittest.main()
