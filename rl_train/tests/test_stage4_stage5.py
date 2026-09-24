from __future__ import annotations

import math
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium import spaces

from rl_train.envs.constant_acceleration_task import build_acceleration_delta_v_task_tuples
from rl_train.envs.myoassist_leg_acceleration import MyoAssistLegAcceleration
from rl_train.envs.myoassist_leg_acceleration_exo import MyoAssistLegAccelerationExo
from rl_train.train.policies.network_index_handler import NetworkIndexHandler


class Stage4Stage5UnitTests(unittest.TestCase):
    CONFIG_DIR = Path("rl_train/train/train_configs")

    @staticmethod
    def _make_fake_sim_with_hip_qvel(right_qvel: float, left_qvel: float):
        joints = {
            "hip_flexion_r": SimpleNamespace(qvel=np.array([right_qvel], dtype=float)),
            "hip_flexion_l": SimpleNamespace(qvel=np.array([left_qvel], dtype=float)),
        }
        return SimpleNamespace(
            data=SimpleNamespace(joint=lambda name: joints[name]),
            model=SimpleNamespace(opt=SimpleNamespace(timestep=1.0 / 1200.0)),
        )

    @staticmethod
    def _make_exo_reward_env(*, executed_action, raw_action=None, previous_action=None):
        env = object.__new__(MyoAssistLegAccelerationExo)
        env.frame_skip = 120
        env._latest_normalized_exo_action = np.asarray(executed_action, dtype=float)
        env._latest_raw_normalized_exo_action = np.asarray(
            raw_action if raw_action is not None else executed_action,
            dtype=float,
        )
        previous = np.asarray(
            previous_action if previous_action is not None else np.zeros(2),
            dtype=float,
        )
        env._exo_action_history = np.concatenate([env._latest_normalized_exo_action, previous, np.zeros(2)])
        env.EXO_JOINT_NAMES = MyoAssistLegAccelerationExo.EXO_JOINT_NAMES
        env._stage5_smat_omega_scale_rad_s = 2.0
        env._stage5_smat_power_alpha = 0.25
        env._stage5_smat_magnitude_beta = 0.2
        env._stage5_smat_saturation_lambda = 2.0
        env._stage5_smat_saturation_delta = 0.8
        env._exo_power_reward_alpha = 0.5
        env._get_desired_power_sign = lambda: 1.0
        env.sim = Stage4Stage5UnitTests._make_fake_sim_with_hip_qvel(1.0, -3.0)
        return env

    def test_network_index_handler_noncontiguous_index(self):
        handler = NetworkIndexHandler(
            {
                "net": {
                    "observation": [
                        {"type": "range", "range": [0, 2]},
                        {"type": "index", "index": 4},
                    ],
                    "action": [{"type": "range_mapping", "range_net": [0, 1], "range_action": [0, 1]}],
                }
            },
            spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        )
        obs = th.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
        mapped = handler.map_observation_to_network(obs, "net")
        self.assertTrue(th.equal(mapped, th.tensor([[10.0, 20.0, 50.0]])))

    def test_stage5_mapping_dimensions(self):
        indexing = {
            "human_actor": {
                "observation": [
                        {"type": "range", "range": [0, 46]},
                        {"type": "index", "index": [52, 53]},
                    ],
                "action": [{"type": "range_mapping", "range_net": [0, 22], "range_action": [0, 22]}],
            },
            "exo_actor": {
                "observation": [
                    {"type": "range", "range": [0, 52]},
                ],
                "action": [{"type": "range_mapping", "range_net": [0, 2], "range_action": [22, 24]}],
            },
            "common_critic": {
                "observation": [{"type": "range", "range": [0, 54]}],
                "action": [],
            },
        }
        handler = NetworkIndexHandler(
            indexing,
            spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32),
        )
        self.assertEqual(handler.get_observation_num("human_actor"), 48)
        self.assertEqual(handler.get_observation_num("exo_actor"), 52)
        self.assertEqual(handler.get_observation_num("common_critic"), 54)

    def test_acceleration_delta_v_task_pool_23_tuples(self):
        tuples = build_acceleration_delta_v_task_tuples(
            acceleration_magnitudes=[0.1, 0.2, 0.3, 0.4, 0.5],
            delta_velocity_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            min_ramp_duration=1.0,
            max_ramp_duration=5.0,
        )
        self.assertEqual(len(tuples), 23)
        self.assertTrue(any(not math.isclose(t.ramp_duration, round(t.ramp_duration)) for t in tuples))
        for task_tuple in tuples:
            self.assertGreaterEqual(task_tuple.ramp_duration + 1e-9, 1.0)
            self.assertLessEqual(task_tuple.ramp_duration - 1e-9, 5.0)

    def test_stage5_smat_reward_helpers(self):
        dt = 0.1
        u_hat = np.array([0.5, -0.9])
        hip_qvel = np.array([1.0, -3.0])
        omega_hat = np.clip(hip_qvel / 2.0, -1.0, 1.0)
        expected = dt * np.sum(
            0.3 * u_hat * omega_hat
            - 0.15 * np.square(u_hat)
            - 2.0 * np.square(np.maximum(0.0, np.abs(u_hat) - 0.8))
        )
        actual = MyoAssistLegAccelerationExo.compute_stage5_smat_power_magnitude_saturation_reward(
            dt=dt,
            normalized_exo_action=u_hat,
            hip_qvel=hip_qvel,
            omega_scale=2.0,
            alpha=0.3,
            beta=0.15,
            saturation_lambda=2.0,
            saturation_delta=0.8,
        )
        self.assertAlmostEqual(actual, float(expected), places=12)

        u_prev = np.array([0.25, -0.1])
        expected_rate = -dt * np.sum(np.square(u_hat - u_prev))
        actual_rate = MyoAssistLegAccelerationExo.compute_stage5_smat_torque_rate_penalty(
            dt=dt,
            normalized_exo_action=u_hat,
            previous_normalized_exo_action=u_prev,
        )
        self.assertAlmostEqual(actual_rate, float(expected_rate), places=12)

    def test_executed_exo_reward_terms_ignore_raw_actor_action_when_lpf_enabled(self):
        executed = np.array([0.25, -0.5])
        raw = np.array([1.0, -1.0])
        previous = np.array([0.1, -0.2])
        env = self._make_exo_reward_env(
            executed_action=executed,
            raw_action=raw,
            previous_action=previous,
        )
        env._get_exo_power = lambda: np.array([2.0, 4.0], dtype=float)

        power_reward = env._calculate_exo_power_reward()
        expected_power = env.dt * np.sum(
            env._exo_power_reward_alpha * np.square(executed) * np.array([1.0, 1.0])
        )
        raw_power = env.dt * np.sum(
            env._exo_power_reward_alpha * np.square(raw) * np.array([1.0, 1.0])
        )
        self.assertAlmostEqual(power_reward, float(expected_power), places=12)
        self.assertNotAlmostEqual(power_reward, float(raw_power), places=12)

        timing_reward = env._calculate_stage5_smat_power_magnitude_saturation_reward()
        hip_qvel = np.array([1.0, -3.0], dtype=float)
        expected_timing = MyoAssistLegAccelerationExo.compute_stage5_smat_power_magnitude_saturation_reward(
            dt=env.dt,
            normalized_exo_action=executed,
            hip_qvel=hip_qvel,
            omega_scale=env._stage5_smat_omega_scale_rad_s,
            alpha=env._stage5_smat_power_alpha,
            beta=env._stage5_smat_magnitude_beta,
            saturation_lambda=env._stage5_smat_saturation_lambda,
            saturation_delta=env._stage5_smat_saturation_delta,
        )
        raw_timing = MyoAssistLegAccelerationExo.compute_stage5_smat_power_magnitude_saturation_reward(
            dt=env.dt,
            normalized_exo_action=raw,
            hip_qvel=hip_qvel,
            omega_scale=env._stage5_smat_omega_scale_rad_s,
            alpha=env._stage5_smat_power_alpha,
            beta=env._stage5_smat_magnitude_beta,
            saturation_lambda=env._stage5_smat_saturation_lambda,
            saturation_delta=env._stage5_smat_saturation_delta,
        )
        self.assertAlmostEqual(timing_reward, expected_timing, places=12)
        self.assertNotAlmostEqual(timing_reward, raw_timing, places=12)

        rate_penalty = env._calculate_stage5_smat_torque_rate_penalty()
        expected_rate = MyoAssistLegAccelerationExo.compute_stage5_smat_torque_rate_penalty(
            dt=env.dt,
            normalized_exo_action=executed,
            previous_normalized_exo_action=previous,
        )
        raw_rate = MyoAssistLegAccelerationExo.compute_stage5_smat_torque_rate_penalty(
            dt=env.dt,
            normalized_exo_action=raw,
            previous_normalized_exo_action=previous,
        )
        self.assertAlmostEqual(rate_penalty, expected_rate, places=12)
        self.assertNotAlmostEqual(rate_penalty, raw_rate, places=12)

    def test_lpf_disabled_executed_reward_terms_match_raw_actor_action(self):
        raw_and_executed = np.array([0.5, -0.9])
        previous = np.array([0.25, -0.1])
        env = self._make_exo_reward_env(
            executed_action=raw_and_executed,
            raw_action=raw_and_executed,
            previous_action=previous,
        )
        env._get_exo_power = lambda: np.array([2.0, -4.0], dtype=float)

        expected_power = env.dt * np.sum(
            env._exo_power_reward_alpha * np.square(raw_and_executed) * np.array([1.0, -1.0])
        )
        self.assertAlmostEqual(env._calculate_exo_power_reward(), float(expected_power), places=12)

        hip_qvel = np.array([1.0, -3.0], dtype=float)
        expected_timing = MyoAssistLegAccelerationExo.compute_stage5_smat_power_magnitude_saturation_reward(
            dt=env.dt,
            normalized_exo_action=raw_and_executed,
            hip_qvel=hip_qvel,
            omega_scale=env._stage5_smat_omega_scale_rad_s,
            alpha=env._stage5_smat_power_alpha,
            beta=env._stage5_smat_magnitude_beta,
            saturation_lambda=env._stage5_smat_saturation_lambda,
            saturation_delta=env._stage5_smat_saturation_delta,
        )
        self.assertAlmostEqual(
            env._calculate_stage5_smat_power_magnitude_saturation_reward(),
            expected_timing,
            places=12,
        )

        expected_rate = MyoAssistLegAccelerationExo.compute_stage5_smat_torque_rate_penalty(
            dt=env.dt,
            normalized_exo_action=raw_and_executed,
            previous_normalized_exo_action=previous,
        )
        self.assertAlmostEqual(env._calculate_stage5_smat_torque_rate_penalty(), expected_rate, places=12)

    def test_exo_output_lpf_formula_and_independent_channels(self):
        dt = 1.0 / 30.0
        tau_s = 0.1
        alpha = MyoAssistLegAccelerationExo.compute_exo_output_lpf_alpha(dt=dt, tau_s=tau_s)
        previous = np.array([0.2, -0.4])
        raw = np.array([1.0, 0.5])
        expected = previous + alpha * (raw - previous)
        actual = MyoAssistLegAccelerationExo.compute_exo_output_lpf(raw, previous, alpha)
        self.assertAlmostEqual(alpha, 0.28346868942621073, places=15)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        self.assertNotAlmostEqual(actual[0], actual[1])

    def test_exo_output_lpf_state_resets_per_episode(self):
        env = object.__new__(MyoAssistLegAccelerationExo)
        env._exo_output_lpf_tau_s = 0.1
        env._exo_output_lpf_alpha_effective = MyoAssistLegAccelerationExo.compute_exo_output_lpf_alpha(
            dt=1.0 / 30.0,
            tau_s=0.1,
        )
        env._exo_output_lpf_state = np.array([0.5, -0.5], dtype=float)
        env._exo_action_history = np.ones(6, dtype=float)
        env._latest_normalized_exo_action = np.ones(2, dtype=float)
        env._latest_raw_normalized_exo_action = np.ones(2, dtype=float)

        env._reset_exo_action_history()

        np.testing.assert_array_equal(env._exo_output_lpf_state, np.zeros(2))
        np.testing.assert_array_equal(env._exo_action_history, np.zeros(6))
        np.testing.assert_array_equal(env._latest_normalized_exo_action, np.zeros(2))
        np.testing.assert_array_equal(env._latest_raw_normalized_exo_action, np.zeros(2))

        filtered = env._apply_exo_output_lpf(np.array([1.0, -1.0]))
        np.testing.assert_allclose(
            filtered,
            np.array([env._exo_output_lpf_alpha_effective, -env._exo_output_lpf_alpha_effective]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_exo_output_lpf_step_execution_chain_disabled_and_enabled(self):
        captured = []

        def fake_parent_step(_env, action, **_kwargs):
            captured.append(action)
            return None, 0.0, False, False, {}

        env = object.__new__(MyoAssistLegAccelerationExo)
        env._exo_action_history = np.zeros(6, dtype=float)
        env._latest_normalized_exo_action = np.zeros(2, dtype=float)
        env._latest_raw_normalized_exo_action = np.zeros(2, dtype=float)
        env._exo_output_lpf_state = np.zeros(2, dtype=float)
        env._exo_output_lpf_tau_s = 0.1
        env._exo_output_lpf_alpha_effective = MyoAssistLegAccelerationExo.compute_exo_output_lpf_alpha(
            dt=1.0 / 30.0,
            tau_s=0.1,
        )
        env._exo_torque_limit_nm = 6.0
        env._torque_limit_reference_nm = 25.0
        env.frame_skip = 40
        env.EXO_JOINT_NAMES = MyoAssistLegAccelerationExo.EXO_JOINT_NAMES
        env._stage5_smat_omega_scale_rad_s = 2.0
        env._stage5_smat_power_alpha = 0.25
        env._stage5_smat_magnitude_beta = 0.2
        env._stage5_smat_saturation_lambda = 2.0
        env._stage5_smat_saturation_delta = 0.8
        env.sim = self._make_fake_sim_with_hip_qvel(2.0, -2.0)
        env.get_exo_teacher_diagnostics = lambda: {}
        action = np.zeros(24, dtype=float)
        action[22:24] = [1.0, -1.0]

        with patch.object(MyoAssistLegAcceleration, "step", fake_parent_step):
            env._exo_output_lpf_enabled = False
            env.step(action)
            self.assertIs(captured[-1], action)
            np.testing.assert_array_equal(captured[-1][22:24], np.array([1.0, -1.0]))

            env._reset_exo_action_history()
            env._exo_output_lpf_enabled = True
            env.step(action)
            self.assertIsNot(captured[-1], action)
            expected = np.array([env._exo_output_lpf_alpha_effective, -env._exo_output_lpf_alpha_effective])
            np.testing.assert_allclose(captured[-1][22:24], expected, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(env._latest_normalized_exo_action, expected, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(env._latest_raw_normalized_exo_action, np.array([1.0, -1.0]), rtol=0.0, atol=1e-12)
            timing_reward = env._calculate_stage5_smat_power_magnitude_saturation_reward()
            expected_timing = MyoAssistLegAccelerationExo.compute_stage5_smat_power_magnitude_saturation_reward(
                dt=env.dt,
                normalized_exo_action=expected,
                hip_qvel=np.array([2.0, -2.0], dtype=float),
                omega_scale=env._stage5_smat_omega_scale_rad_s,
                alpha=env._stage5_smat_power_alpha,
                beta=env._stage5_smat_magnitude_beta,
                saturation_lambda=env._stage5_smat_saturation_lambda,
                saturation_delta=env._stage5_smat_saturation_delta,
            )
            self.assertAlmostEqual(timing_reward, expected_timing, places=12)

    def test_exo_output_lpf_rejects_enabled_deprecated_alpha_without_tau(self):
        env = object.__new__(MyoAssistLegAccelerationExo)
        env._exo_output_lpf_enabled = True
        env._exo_output_lpf_tau_s = None
        with self.assertRaisesRegex(ValueError, "exo_output_lpf_alpha is deprecated"):
            env._warn_or_reject_deprecated_lpf_alpha(
                type("Params", (), {"exo_output_lpf_alpha": 0.30})()
            )

    def test_s4_s5_configs_include_disabled_exo_output_lpf_defaults(self):
        config_names = (
            "exo_phase_1_steady.json",
            "exo_phase_1_ac.json",
            "exo_phase_1_de.json",
            "exo_phase_2_steady.json",
            "exo_phase_2_ac.json",
            "exo_phase_2_de.json",
        )
        for config_name in config_names:
            with self.subTest(config_name=config_name):
                with (self.CONFIG_DIR / config_name).open("r", encoding="utf-8") as f:
                    env_params = json.load(f)["env_params"]
                self.assertIn("exo_output_lpf_enabled", env_params)
                self.assertIn("exo_output_lpf_tau_s", env_params)
                self.assertNotIn("exo_output_lpf_alpha", env_params)
                self.assertIs(env_params["exo_output_lpf_enabled"], False)
                self.assertAlmostEqual(env_params["exo_output_lpf_tau_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
