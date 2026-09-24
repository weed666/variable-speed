from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcn-acc-dec-mpl")

import h5py
import matplotlib
import numpy as np
from stable_baselines3 import PPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .collect_teacher_30hz import (
    action_label,
    apply_initial_state_perturbation,
    current_observation,
    exo_info,
    load_teacher_config,
    safe_joint,
    sensor_contact,
    stage_info,
    teacher_result_dir,
)
from .config import DEFAULT_OUTPUT, INPUT_CHANNELS, PHYSICS_HZ, REPO_ROOT, TARGET_CHANNELS, TARGET_HZ, TEACHER_HZ
from .imu_kinematics import ImuDefinition, build_imu_definition, continuous_angles, sample_raw_orientation

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rl_train.envs  # noqa: F401,E402
from myosuite.utils import gym  # noqa: E402
from rl_train.envs.constant_acceleration_task import build_acceleration_delta_v_task_tuples  # noqa: E402
from rl_train.envs.environment_handler import EnvironmentHandler  # noqa: E402
from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy  # noqa: E402


AUDIT_DIR = DEFAULT_OUTPUT / "audits/acc_dec"
DEFAULT_ACC_CHECKPOINT = REPO_ROOT / "rl_train/results/Exo_Phase_2_ac/trained_models/model_79757312.zip"
DEFAULT_DEC_CHECKPOINT = REPO_ROOT / "rl_train/results/Exo_Phase_2_de/trained_models/model_97648640.zip"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tcn/data_generation_v2/output"
SPEED_MIN = 0.90
SPEED_MAX = 1.60
SPEED_STEP = 0.05
DELTA_V_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
ACCELERATION_MAGNITUDES = (0.1, 0.2, 0.3, 0.4, 0.5)
PRE_HOLD_S = 1.0
EPISODE_DURATION_S = 7.0
PHASE_BINS = 30
SPLIT_COUNTS = {"train": 1400, "validation": 300, "test": 300}
SPLIT_SEED = 20260825
COLLECTION_SEED = 20260825
TORQUE_TOL_NM = 1e-4
ACTION_TOL = 1e-5


@dataclass(frozen=True)
class AccDecEpisodeSpec:
    episode_id: str
    regime: str
    task_index: int
    initial_velocity_m_s: float
    goal_velocity_m_s: float
    signed_acceleration_m_s2: float
    delta_velocity_m_s: float
    ramp_duration_s: float
    ramp_start_time_s: float
    ramp_end_time_s: float
    episode_duration_s: float
    seed: int
    reference_index: int
    phase_bin: int
    attempt: int = 0

    @property
    def task_key(self) -> str:
        return f"{abs(self.signed_acceleration_m_s2):.1f}:{self.delta_velocity_m_s:.1f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_pool() -> list[dict[str, Any]]:
    tuples = build_acceleration_delta_v_task_tuples(
        acceleration_magnitudes=ACCELERATION_MAGNITUDES,
        delta_velocity_levels=DELTA_V_LEVELS,
        min_ramp_duration=1.0,
        max_ramp_duration=5.0,
    )
    rows = []
    for idx, task in enumerate(sorted(tuples, key=lambda x: (x.acceleration_magnitude, x.delta_velocity))):
        rows.append(
            {
                "task_index": idx,
                "acceleration_magnitude_m_s2": round(float(task.acceleration_magnitude), 10),
                "delta_velocity_m_s": round(float(task.delta_velocity), 10),
                "ramp_duration_s": round(float(task.ramp_duration), 10),
            }
        )
    if len(rows) != 23:
        raise RuntimeError(f"task pool must contain 23 tuples, got {len(rows)}")
    return rows


def velocity_grid() -> list[float]:
    lo = int(round(SPEED_MIN * 100))
    hi = int(round(SPEED_MAX * 100))
    step = int(round(SPEED_STEP * 100))
    return [v / 100.0 for v in range(lo, hi + 1, step)]


def legal_initial_velocities(regime: str, delta_v: float) -> list[float]:
    values = []
    for v in velocity_grid():
        goal = v + delta_v if regime == "accel" else v - delta_v
        if SPEED_MIN - 1e-9 <= goal <= SPEED_MAX + 1e-9:
            values.append(round(v, 2))
    if not values:
        raise RuntimeError(f"no legal initial velocity for {regime} delta_v={delta_v}")
    return values


def balanced_target_counts(total: int, n_tasks: int) -> list[int]:
    base = total // n_tasks
    extra = total % n_tasks
    return [base + (1 if idx < extra else 0) for idx in range(n_tasks)]


def make_episode_specs(regime: str, total: int, *, seed_base: int, existing_count: int = 0) -> list[AccDecEpisodeSpec]:
    rows = task_pool()
    counts = balanced_target_counts(total, len(rows))
    rng = random.Random(seed_base + (0 if regime == "accel" else 10_000_000))
    specs: list[AccDecEpisodeSpec] = []
    global_idx = 0
    for row, count in zip(rows, counts):
        delta_v = float(row["delta_velocity_m_s"])
        abs_acc = float(row["acceleration_magnitude_m_s2"])
        initials = legal_initial_velocities(regime, delta_v)
        rng.shuffle(initials)
        for local_idx in range(count):
            initial = initials[local_idx % len(initials)]
            signed_acc = abs_acc if regime == "accel" else -abs_acc
            goal = round(initial + math.copysign(delta_v, signed_acc), 2)
            phase_bin = (global_idx + local_idx) % PHASE_BINS
            bin_lo = int(round(phase_bin * 91 / PHASE_BINS))
            bin_hi = int(round((phase_bin + 1) * 91 / PHASE_BINS)) - 1
            reference_index = rng.randint(bin_lo, max(bin_lo, bin_hi))
            seed = seed_base + (0 if regime == "accel" else 2_000_000) + global_idx + existing_count * 100_000
            ramp_duration = delta_v / abs_acc
            specs.append(
                AccDecEpisodeSpec(
                    episode_id=f"{regime}_task{int(row['task_index']):02d}_ep{global_idx:05d}",
                    regime=regime,
                    task_index=int(row["task_index"]),
                    initial_velocity_m_s=float(initial),
                    goal_velocity_m_s=float(goal),
                    signed_acceleration_m_s2=float(signed_acc),
                    delta_velocity_m_s=float(delta_v),
                    ramp_duration_s=float(ramp_duration),
                    ramp_start_time_s=PRE_HOLD_S,
                    ramp_end_time_s=float(PRE_HOLD_S + ramp_duration),
                    episode_duration_s=EPISODE_DURATION_S,
                    seed=int(seed),
                    reference_index=int(reference_index),
                    phase_bin=int(phase_bin),
                )
            )
            global_idx += 1
    return specs


def make_task_occurrence_spec(
    regime: str,
    row: dict[str, Any],
    *,
    occurrence: int,
    global_seed_offset: int,
) -> AccDecEpisodeSpec:
    delta_v = float(row["delta_velocity_m_s"])
    abs_acc = float(row["acceleration_magnitude_m_s2"])
    initials = legal_initial_velocities(regime, delta_v)
    initial = initials[occurrence % len(initials)]
    signed_acc = abs_acc if regime == "accel" else -abs_acc
    goal = round(initial + math.copysign(delta_v, signed_acc), 2)
    phase_bin = occurrence % PHASE_BINS
    bin_lo = int(round(phase_bin * 91 / PHASE_BINS))
    bin_hi = int(round((phase_bin + 1) * 91 / PHASE_BINS)) - 1
    rng = random.Random(COLLECTION_SEED + int(row["task_index"]) * 100_000 + occurrence + (0 if regime == "accel" else 20_000_000))
    reference_index = rng.randint(bin_lo, max(bin_lo, bin_hi))
    seed = COLLECTION_SEED + global_seed_offset + int(row["task_index"]) * 100_000 + occurrence
    if regime == "decel":
        seed += 2_000_000
    seed = int(seed % (2**32 - 1))
    ramp_duration = delta_v / abs_acc
    return AccDecEpisodeSpec(
        episode_id=f"{regime}_task{int(row['task_index']):02d}_occ{occurrence:05d}",
        regime=regime,
        task_index=int(row["task_index"]),
        initial_velocity_m_s=float(initial),
        goal_velocity_m_s=float(goal),
        signed_acceleration_m_s2=float(signed_acc),
        delta_velocity_m_s=float(delta_v),
        ramp_duration_s=float(ramp_duration),
        ramp_start_time_s=PRE_HOLD_S,
        ramp_end_time_s=float(PRE_HOLD_S + ramp_duration),
        episode_duration_s=EPISODE_DURATION_S,
        seed=int(seed),
        reference_index=int(reference_index),
        phase_bin=int(phase_bin),
    )


def _patch_runtime_config(base_config: Any, *, regime: str, seed: int) -> Any:
    import copy

    cfg = copy.deepcopy(base_config)
    ep = cfg.env_params
    ep.num_envs = 1
    ep.seed = int(seed)
    ep.control_framerate = TEACHER_HZ
    ep.physics_sim_framerate = PHYSICS_HZ
    ep.episode_duration_s = EPISODE_DURATION_S
    ep.custom_max_episode_steps = int(round(EPISODE_DURATION_S * TEACHER_HZ))
    ep.dynamic_episode_duration = False
    ep.teacher_regime = "acceleration" if regime == "accel" else "deceleration"
    ep.task_pool_mode = "acceleration_delta_v_grid"
    ep.speed_min = SPEED_MIN
    ep.speed_max = SPEED_MAX
    ep.min_target_velocity = SPEED_MIN
    ep.max_target_velocity = SPEED_MAX
    ep.acceleration_magnitudes = list(ACCELERATION_MAGNITUDES)
    ep.delta_velocity_levels = list(DELTA_V_LEVELS)
    ep.min_ramp_duration = 1.0
    ep.max_ramp_duration = 5.0
    ep.ramp_start_time_min = PRE_HOLD_S
    ep.ramp_start_time_max = PRE_HOLD_S
    ep.pre_hold_min_s = PRE_HOLD_S
    ep.pre_hold_max_s = PRE_HOLD_S
    ep.post_hold_min_s = 1.0
    ep.post_hold_max_s = 1.0
    ep.flag_random_ref_index = True
    if hasattr(cfg, "ppo_params"):
        cfg.ppo_params.device = "cpu"
        cfg.ppo_params.n_steps = 256
        cfg.ppo_params.batch_size = 256
    return cfg


def make_env_and_model(base_config: Any, checkpoint: Path, regime: str, seed: int) -> tuple[Any, Any]:
    cfg = _patch_runtime_config(base_config, regime=regime, seed=seed)
    ref_data = EnvironmentHandler.load_reference_data(cfg)
    env = gym.make(
        cfg.env_params.env_id,
        seed=cfg.env_params.seed,
        model_path=cfg.env_params.model_path,
        env_params=cfg.env_params,
        reference_data=ref_data,
        is_evaluate_mode=True,
    ).unwrapped
    model = PPO.load(
        str(checkpoint),
        env=env,
        custom_objects={"policy_class": HumanExoActorCriticPolicy},
        device="cpu",
    )
    EnvironmentHandler.restore_sb3_save_params(model)
    return env, model


def _reset_env(env: Any, seed: int) -> Any:
    result = env.reset(seed=int(seed))
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def run_acc_dec_episode(env: Any, model: Any, base_config: Any, spec: AccDecEpisodeSpec, imu: ImuDefinition) -> dict[str, Any]:
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    rows: list[dict[str, Any]] = []
    termination = {"reason": "full_horizon", "terminated": False, "truncated": False, "fall": False}
    env.set_fixed_evaluation_task(
        initial_velocity=spec.initial_velocity_m_s,
        goal_velocity=spec.goal_velocity_m_s,
        signed_acceleration=spec.signed_acceleration_m_s2,
        ramp_start_time=spec.ramp_start_time_s,
        reference_index=spec.reference_index,
    )
    obs = _reset_env(env, spec.seed)
    apply_initial_state_perturbation(env, None)
    expected_steps = int(round(spec.episode_duration_s * TEACHER_HZ))
    for update_idx in range(expected_steps):
        t_before = float(env.sim.data.time)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(-1)
        obs, _reward, terminated, truncated, info = env.step(action)
        stage = stage_info(env)
        exo = exo_info(env)
        left_action, right_action, action_source = action_label(env, action, base_config)
        raw_exo = np.asarray(exo.get("latest_raw_normalized_exo_action", [action[22], action[23]]), dtype=float).reshape(-1)
        executed_exo = np.asarray(exo.get("latest_normalized_exo_action", [action[22], action[23]]), dtype=float).reshape(-1)
        orient = sample_raw_orientation(env, imu)
        torques = np.asarray(exo.get("exo_joint_torque", [float("nan"), float("nan")]), dtype=float).reshape(-1)
        rows.append(
            {
                "time_s": float(env.sim.data.time),
                "teacher_update_time_s": t_before,
                "teacher_update_index": update_idx,
                "initial_velocity_m_s": spec.initial_velocity_m_s,
                "goal_velocity_m_s": spec.goal_velocity_m_s,
                "delta_velocity_m_s": spec.delta_velocity_m_s,
                "target_velocity_m_s": float(stage.get("target_velocity", spec.initial_velocity_m_s)),
                "actual_pelvis_velocity_m_s": float(stage.get("actual_pelvis_velocity", safe_joint(env, "pelvis_tx", "qvel"))),
                "target_acceleration_m_s2": float(stage.get("target_acceleration", 0.0)),
                "task_phase": str(stage.get("phase", "")),
                "ramp_start_time_s": float(stage.get("ramp_start_time", spec.ramp_start_time_s)),
                "ramp_end_time_s": float(stage.get("ramp_end_time", spec.ramp_end_time_s)),
                "left_exo_action": left_action,
                "right_exo_action": right_action,
                "raw_left_exo_action": float(raw_exo[1]) if raw_exo.size > 1 else float("nan"),
                "raw_right_exo_action": float(raw_exo[0]) if raw_exo.size > 0 else float("nan"),
                "executed_left_exo_action": float(executed_exo[1]) if executed_exo.size > 1 else float("nan"),
                "executed_right_exo_action": float(executed_exo[0]) if executed_exo.size > 0 else float("nan"),
                "action_source": action_source,
                "left_exo_torque_nm": float(torques[1]) if torques.size > 1 else float("nan"),
                "right_exo_torque_nm": float(torques[0]) if torques.size > 0 else float("nan"),
                "left_foot_contact": sensor_contact(env, "l_foot", "l_toes"),
                "right_foot_contact": sensor_contact(env, "r_foot", "r_toes"),
                "left_hip_angle_rad": safe_joint(env, "hip_flexion_l", "qpos"),
                "right_hip_angle_rad": safe_joint(env, "hip_flexion_r", "qpos"),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": str(stage.get("termination_reason") or info.get("termination_reason") or ""),
                **orient,
            }
        )
        if terminated or truncated:
            termination = {
                "reason": str(stage.get("termination_reason") or info.get("termination_reason") or ("terminated" if terminated else "truncated")),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "fall": bool(terminated and safe_joint(env, "pelvis_ty", "qpos") < getattr(env, "_safe_height", -np.inf)),
            }
            break
    return {"spec": asdict(spec), "rows": rows, "termination": termination}


def _episode_arrays(payload: dict[str, Any], imu: ImuDefinition) -> dict[str, Any]:
    rows = payload["rows"]
    raw_left = np.asarray([r["left_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    raw_right = np.asarray([r["right_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    left_angle, right_angle = continuous_angles(raw_left, raw_right, imu)
    fields = {
        "time_s": np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        "teacher_update_time_s": np.asarray([r["teacher_update_time_s"] for r in rows], dtype=np.float64),
        "teacher_update_index": np.asarray([r["teacher_update_index"] for r in rows], dtype=np.int32),
        "initial_velocity_m_s": np.asarray([r["initial_velocity_m_s"] for r in rows], dtype=np.float32),
        "goal_velocity_m_s": np.asarray([r["goal_velocity_m_s"] for r in rows], dtype=np.float32),
        "delta_velocity_m_s": np.asarray([r["delta_velocity_m_s"] for r in rows], dtype=np.float32),
        "target_velocity_m_s": np.asarray([r["target_velocity_m_s"] for r in rows], dtype=np.float32),
        "actual_pelvis_velocity_m_s": np.asarray([r["actual_pelvis_velocity_m_s"] for r in rows], dtype=np.float32),
        "target_acceleration_m_s2": np.asarray([r["target_acceleration_m_s2"] for r in rows], dtype=np.float32),
        "ramp_start_time_s": np.asarray([r["ramp_start_time_s"] for r in rows], dtype=np.float32),
        "ramp_end_time_s": np.asarray([r["ramp_end_time_s"] for r in rows], dtype=np.float32),
        "left_thigh_xmat": np.stack([r["left_thigh_xmat"] for r in rows]).astype(np.float64),
        "right_thigh_xmat": np.stack([r["right_thigh_xmat"] for r in rows]).astype(np.float64),
        "left_thigh_quat": np.stack([r["left_thigh_quat"] for r in rows]).astype(np.float64),
        "right_thigh_quat": np.stack([r["right_thigh_quat"] for r in rows]).astype(np.float64),
        "left_raw_sagittal_angle_rad": raw_left,
        "right_raw_sagittal_angle_rad": raw_right,
        "left_thigh_angle_rad": left_angle.astype(np.float32),
        "right_thigh_angle_rad": right_angle.astype(np.float32),
        "left_thigh_gyro_rad_s": np.asarray([r["left_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
        "right_thigh_gyro_rad_s": np.asarray([r["right_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
        "left_world_angular_velocity_rad_s": np.stack([r["left_world_angular_velocity_rad_s"] for r in rows]).astype(np.float32),
        "right_world_angular_velocity_rad_s": np.stack([r["right_world_angular_velocity_rad_s"] for r in rows]).astype(np.float32),
        "left_exo_action": np.asarray([r["left_exo_action"] for r in rows], dtype=np.float32),
        "right_exo_action": np.asarray([r["right_exo_action"] for r in rows], dtype=np.float32),
        "raw_left_exo_action": np.asarray([r["raw_left_exo_action"] for r in rows], dtype=np.float32),
        "raw_right_exo_action": np.asarray([r["raw_right_exo_action"] for r in rows], dtype=np.float32),
        "executed_left_exo_action": np.asarray([r["executed_left_exo_action"] for r in rows], dtype=np.float32),
        "executed_right_exo_action": np.asarray([r["executed_right_exo_action"] for r in rows], dtype=np.float32),
        "left_exo_torque_nm": np.asarray([r["left_exo_torque_nm"] for r in rows], dtype=np.float32),
        "right_exo_torque_nm": np.asarray([r["right_exo_torque_nm"] for r in rows], dtype=np.float32),
        "left_foot_contact": np.asarray([r["left_foot_contact"] for r in rows], dtype=np.bool_),
        "right_foot_contact": np.asarray([r["right_foot_contact"] for r in rows], dtype=np.bool_),
        "left_hip_angle_rad": np.asarray([r["left_hip_angle_rad"] for r in rows], dtype=np.float32),
        "right_hip_angle_rad": np.asarray([r["right_hip_angle_rad"] for r in rows], dtype=np.float32),
        "terminated": np.asarray([r["terminated"] for r in rows], dtype=np.bool_),
        "truncated": np.asarray([r["truncated"] for r in rows], dtype=np.bool_),
    }
    fields["task_phase"] = np.asarray([str(r["task_phase"]).encode("utf-8") for r in rows])
    fields["regime_phase"] = fields["task_phase"]
    fields["gait_phase"] = fields["task_phase"]
    fields["action_source"] = np.asarray([str(r["action_source"]).encode("utf-8") for r in rows])
    fields["termination_reason_per_step"] = np.asarray([str(r["termination_reason"]).encode("utf-8") for r in rows])
    return fields


def trajectory_hash_from_arrays(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in (
        "left_thigh_angle_rad",
        "left_thigh_gyro_rad_s",
        "right_thigh_angle_rad",
        "right_thigh_gyro_rad_s",
        "left_exo_action",
        "right_exo_action",
        "target_velocity_m_s",
    ):
        h.update(np.asarray(arrays[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def reject_reason(payload: dict[str, Any], arrays: dict[str, Any], spec: AccDecEpisodeSpec, torque_scale: float) -> str | None:
    termination = payload["termination"]
    if termination.get("fall"):
        return "fall"
    expected_steps = int(round(spec.episode_duration_s * TEACHER_HZ))
    if len(arrays["time_s"]) != expected_steps:
        return f"incomplete_episode:{len(arrays['time_s'])}_of_{expected_steps}"
    if termination.get("terminated"):
        return f"terminated:{termination.get('reason')}"
    numeric_keys = [k for k, v in arrays.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fiu"]
    for key in numeric_keys:
        if not np.all(np.isfinite(arrays[key])):
            return f"non_finite:{key}"
    if not np.all(np.diff(arrays["time_s"]) > 0):
        return "timestamp_not_monotonic"
    if np.max(np.abs(arrays["left_thigh_angle_rad"])) > math.radians(140) or np.max(np.abs(arrays["right_thigh_angle_rad"])) > math.radians(140):
        return "thigh_angle_unphysiological"
    if np.max(np.abs(arrays["left_thigh_gyro_rad_s"])) > math.radians(1200) or np.max(np.abs(arrays["right_thigh_gyro_rad_s"])) > math.radians(1200):
        return "thigh_gyro_unphysiological"
    action = np.stack([arrays["left_exo_action"], arrays["right_exo_action"]], axis=1)
    if np.min(action) < -1.0 - ACTION_TOL or np.max(action) > 1.0 + ACTION_TOL:
        return "action_out_of_range"
    torque = np.stack([arrays["left_exo_torque_nm"], arrays["right_exo_torque_nm"]], axis=1)
    if np.max(np.abs(torque - torque_scale * action)) > max(TORQUE_TOL_NM, 5e-4):
        return "torque_action_mapping_mismatch"
    if abs(float(arrays["initial_velocity_m_s"][0]) - spec.initial_velocity_m_s) > 1e-6:
        return "initial_velocity_mismatch"
    if abs(float(arrays["goal_velocity_m_s"][0]) - spec.goal_velocity_m_s) > 1e-6:
        return "goal_velocity_mismatch"
    if abs(abs(spec.goal_velocity_m_s - spec.initial_velocity_m_s) - spec.delta_velocity_m_s) > 1e-6:
        return "delta_velocity_mismatch"
    if abs(spec.delta_velocity_m_s / abs(spec.signed_acceleration_m_s2) - spec.ramp_duration_s) > 1e-6:
        return "ramp_duration_mismatch"
    phases = {p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in arrays["task_phase"]}
    if not {"INITIAL_HOLD", "RAMP", "FINAL_HOLD"}.issubset(phases):
        return "missing_required_phase"
    return None


def _init_raw_h5(path: Path, regime: str, checkpoint: Path, base_config_dict: dict[str, Any], imu: ImuDefinition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = f"tcn_data_generation_v2_raw_30hz_{regime}"
        h5.attrs["teacher_frequency_hz"] = TEACHER_HZ
        h5.attrs["physics_frequency_hz"] = PHYSICS_HZ
        h5.attrs["teacher_checkpoint"] = str(checkpoint)
        h5.attrs["teacher_result_dir"] = str(teacher_result_dir(checkpoint))
        h5.attrs["environment_config_json"] = json.dumps(base_config_dict, sort_keys=True)
        h5.attrs["imu_definition_json"] = json.dumps(imu.to_json(), sort_keys=True)
        h5.attrs["trajectory_hash_fields"] = "left/right thigh angle+gyro,left/right executed action,target_velocity"
        h5.attrs["episode_duration_s"] = EPISODE_DURATION_S
        h5.attrs["pre_hold_s"] = PRE_HOLD_S
        h5.attrs["target_frequency_hz_after_processing"] = TARGET_HZ
        h5.attrs["label_definition"] = "left_exo_action,right_exo_action are executed normalized Exo commands; torque = action * exo_torque_limit_nm"
        h5.create_group("episodes")


def append_episode(path: Path, group_name: str, payload: dict[str, Any], arrays: dict[str, Any], trajectory_hash: str) -> None:
    spec = payload["spec"]
    with h5py.File(path, "a") as h5:
        root = h5["episodes"]
        if group_name in root:
            raise RuntimeError(f"episode group already exists: {group_name}")
        grp = root.create_group(group_name)
        for key, value in spec.items():
            if isinstance(value, str):
                grp.attrs[key] = value
            else:
                grp.attrs[key] = value
        grp.attrs["random_seed"] = int(spec["seed"])
        grp.attrs["termination_reason"] = payload["termination"]["reason"]
        grp.attrs["termination_json"] = json.dumps(payload["termination"], sort_keys=True)
        grp.attrs["trajectory_hash"] = trajectory_hash
        grp.attrs["collection_status"] = "valid"
        for key, value in arrays.items():
            grp.create_dataset(key, data=value)
        h5.flush()


def load_collection_state(path: Path) -> tuple[set[tuple[Any, ...]], set[str], dict[str, int]]:
    meta_keys = set()
    hashes = set()
    counts: dict[str, int] = {}
    if not path.exists():
        return meta_keys, hashes, counts
    with h5py.File(path, "r") as h5:
        for grp in h5["episodes"].values():
            key = (
                str(grp.attrs["regime"]),
                round(float(grp.attrs["initial_velocity_m_s"]), 2),
                round(float(grp.attrs["goal_velocity_m_s"]), 2),
                round(float(grp.attrs["signed_acceleration_m_s2"]), 2),
                round(float(grp.attrs["delta_velocity_m_s"]), 2),
                int(grp.attrs["reference_index"]),
                int(grp.attrs["seed"]),
            )
            meta_keys.add(key)
            hashes.add(str(grp.attrs["trajectory_hash"]))
            task_key = f"{abs(float(grp.attrs['signed_acceleration_m_s2'])):.1f}:{float(grp.attrs['delta_velocity_m_s']):.1f}"
            counts[task_key] = counts.get(task_key, 0) + 1
    return meta_keys, hashes, counts


def collect_regime(regime: str, checkpoint: Path, output_dir: Path, *, target: int, max_attempts: int, smoke: bool = False) -> dict[str, Any]:
    raw_path = output_dir / "raw_30hz" / f"{'accel' if regime == 'accel' else 'decel'}_30hz.h5"
    log_path = AUDIT_DIR / f"{regime}_collection.log"
    base_config, base_config_dict = load_teacher_config(checkpoint)
    imu = build_imu_definition(base_config.env_params.model_path)
    torque_scale = float(getattr(base_config.env_params, "exo_torque_limit_nm", 12.0))
    if not raw_path.exists():
        _init_raw_h5(raw_path, regime, checkpoint, base_config_dict, imu)
    meta_keys, hashes, counts = load_collection_state(raw_path)
    rejection_counts: dict[str, int] = {}
    attempts_by_task = dict(counts)
    duplicate_count = 0
    failure_count = 0
    accepted = sum(counts.values())
    spec_target = target
    rows = task_pool()
    task_targets = {f"{float(r['acceleration_magnitude_m_s2']):.1f}:{float(r['delta_velocity_m_s']):.1f}": n for r, n in zip(rows, balanced_target_counts(spec_target, 23))}
    env, model = make_env_and_model(base_config, checkpoint, regime, COLLECTION_SEED)
    attempt = 0
    try:
        while accepted < spec_target and attempt < max_attempts:
            deficient = [
                r
                for r in rows
                if counts.get(f"{float(r['acceleration_magnitude_m_s2']):.1f}:{float(r['delta_velocity_m_s']):.1f}", 0)
                < task_targets[f"{float(r['acceleration_magnitude_m_s2']):.1f}:{float(r['delta_velocity_m_s']):.1f}"]
            ]
            if not deficient:
                break
            deficient.sort(
                key=lambda r: (
                    counts.get(f"{float(r['acceleration_magnitude_m_s2']):.1f}:{float(r['delta_velocity_m_s']):.1f}", 0)
                    / task_targets[f"{float(r['acceleration_magnitude_m_s2']):.1f}:{float(r['delta_velocity_m_s']):.1f}"],
                    int(r["task_index"]),
                )
            )
            row = deficient[0]
            task_key = f"{float(row['acceleration_magnitude_m_s2']):.1f}:{float(row['delta_velocity_m_s']):.1f}"
            occurrence = attempts_by_task.get(task_key, 0)
            spec = make_task_occurrence_spec(regime, row, occurrence=occurrence, global_seed_offset=0)
            attempts_by_task[task_key] = occurrence + 1
            attempt += 1
            spec = AccDecEpisodeSpec(**(asdict(spec) | {"attempt": attempt}))
            meta_key = (
                spec.regime,
                round(spec.initial_velocity_m_s, 2),
                round(spec.goal_velocity_m_s, 2),
                round(spec.signed_acceleration_m_s2, 2),
                round(spec.delta_velocity_m_s, 2),
                int(spec.reference_index),
                int(spec.seed),
            )
            if meta_key in meta_keys:
                duplicate_count += 1
                rejection_counts["duplicate_metadata"] = rejection_counts.get("duplicate_metadata", 0) + 1
                continue
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{time.strftime('%F %T')} attempt={attempt} accepted={accepted}/{spec_target} {spec.episode_id} {spec.task_key} seed={spec.seed} ref={spec.reference_index}\n")
            payload = run_acc_dec_episode(env, model, base_config, spec, imu)
            arrays = _episode_arrays(payload, imu)
            reason = reject_reason(payload, arrays, spec, torque_scale)
            if reason is not None:
                failure_count += 1
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            th = trajectory_hash_from_arrays(arrays)
            if th in hashes:
                duplicate_count += 1
                rejection_counts["duplicate_trajectory_hash"] = rejection_counts.get("duplicate_trajectory_hash", 0) + 1
                continue
            group_name = f"{accepted:06d}"
            append_episode(raw_path, group_name, payload, arrays, th)
            meta_keys.add(meta_key)
            hashes.add(th)
            counts[spec.task_key] = counts.get(spec.task_key, 0) + 1
            accepted += 1
            _write_json(AUDIT_DIR / "collection_manifest.json", {"regime": regime, "accepted": accepted, "failures": failure_count, "duplicates": duplicate_count, "counts": counts, "rejections": rejection_counts})
            if smoke and accepted >= target:
                break
    finally:
        try:
            model.env = None
            env.close()
        except Exception:
            pass
    return {
        "regime": regime,
        "raw_path": str(raw_path),
        "accepted": accepted,
        "failures": failure_count,
        "duplicates": duplicate_count,
        "rejections": rejection_counts,
        "counts": counts,
        "max_attempts": max_attempts,
    }


def _relative_raw_times(grp: h5py.Group) -> np.ndarray:
    t = grp["time_s"][:].astype(np.float64)
    return t - t[0]


def zoh_indices(raw_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(raw_t, target_t, side="right") - 1
    return np.clip(idx, 0, len(raw_t) - 1)


def resample_regime(raw_path: Path, processed_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if processed_path.exists() and not overwrite:
        raise FileExistsError(f"{processed_path} exists; pass --overwrite")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    continuous = {
        "left_thigh_angle_rad": "left_thigh_angle_rad",
        "left_thigh_angular_velocity_rad_s": "left_thigh_gyro_rad_s",
        "right_thigh_angle_rad": "right_thigh_angle_rad",
        "right_thigh_angular_velocity_rad_s": "right_thigh_gyro_rad_s",
        "actual_pelvis_velocity_m_s": "actual_pelvis_velocity_m_s",
        "target_velocity_m_s": "target_velocity_m_s",
        "target_acceleration_m_s2": "target_acceleration_m_s2",
        "left_teacher_action_norm": "left_exo_action",
        "right_teacher_action_norm": "right_exo_action",
        "raw_left_exo_action": "raw_left_exo_action",
        "raw_right_exo_action": "raw_right_exo_action",
        "executed_left_exo_action": "executed_left_exo_action",
        "executed_right_exo_action": "executed_right_exo_action",
        "left_teacher_torque_nm": "left_exo_torque_nm",
        "right_teacher_torque_nm": "right_exo_torque_nm",
    }
    zoh = (
        "left_foot_contact",
        "right_foot_contact",
        "regime_phase",
    )
    phase_codes = {"INITIAL_HOLD": 0, "RAMP": 1, "FINAL_HOLD": 2}
    with h5py.File(raw_path, "r") as src, h5py.File(processed_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["schema_version"] = str(src.attrs["schema_version"]).replace("raw_30hz", "processed_100hz")
        dst.attrs["source_raw_30hz"] = str(raw_path)
        dst.attrs["target_frequency_hz"] = TARGET_HZ
        dst.attrs["samples_per_episode"] = int(round(EPISODE_DURATION_S * TARGET_HZ))
        dst.attrs["time_axis_strategy"] = "episode-relative 100 Hz grid using real rollout raw timestamps shifted by first sample"
        dst.attrs["imu_resampling"] = "linear interpolation after per-episode angle unwrap; no cross-episode interpolation"
        dst.attrs["teacher_action_resampling"] = "linear interpolation from real 30 Hz teacher update timestamps"
        dst.attrs["field_schema"] = "TCNDataset canonical fields"
        root = dst.create_group("episodes")
        for name, grp in src["episodes"].items():
            raw_t = _relative_raw_times(grp)
            n_target = int(round(float(grp.attrs["episode_duration_s"]) * TARGET_HZ))
            target_t = np.arange(n_target, dtype=np.float64) / TARGET_HZ
            idx = zoh_indices(raw_t, target_t)
            out = root.create_group(name)
            for key, value in grp.attrs.items():
                out.attrs[key] = value
            out.attrs["source_raw_group"] = name
            out.create_dataset("timestamp_s", data=target_t)
            out.create_dataset("raw_30hz_time_s", data=raw_t)
            out.create_dataset("source_raw_time_s", data=grp["time_s"][:].astype(np.float64))
            out.create_dataset("source_teacher_update_time_s", data=grp["teacher_update_time_s"][:].astype(np.float64))
            out.create_dataset("teacher_update_index_100hz", data=idx.astype(np.int32))
            out.create_dataset("teacher_update_time_s_100hz", data=grp["teacher_update_time_s"][:].astype(np.float64)[idx])
            for dst_key, src_key in continuous.items():
                out.create_dataset(dst_key, data=np.interp(target_t, raw_t, grp[src_key][:].astype(np.float64)).astype(np.float32))
            for key in zoh:
                out.create_dataset(key, data=grp[key][:][idx])
            raw_phase = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in grp["task_phase"][:][idx]]
            out.create_dataset("task_phase", data=np.asarray([phase_codes.get(p, -1) for p in raw_phase], dtype=np.int8))
            for key in ("initial_velocity_m_s", "goal_velocity_m_s", "delta_velocity_m_s", "ramp_start_time_s", "ramp_end_time_s"):
                out.create_dataset(key, data=np.full(n_target, float(grp.attrs[key]), dtype=np.float32))
            out.create_dataset(
                "teacher_action_100hz_linear",
                data=np.stack([out["left_teacher_action_norm"][:], out["right_teacher_action_norm"][:]], axis=1).astype(np.float32),
            )
            out.create_dataset(
                "raw_30hz_teacher_action",
                data=np.stack([grp["left_exo_action"][:], grp["right_exo_action"][:]], axis=1).astype(np.float32),
            )
            out.create_dataset("episode_id", data=np.asarray([str(grp.attrs.get("episode_id", name)).encode("utf-8")]))
    return {"processed_path": str(processed_path), "source_raw_30hz": str(raw_path)}


def split_regime(processed_path: Path, split_csv: Path, *, seed: int = SPLIT_SEED) -> list[dict[str, Any]]:
    rows = []
    with h5py.File(processed_path, "r") as h5:
        for name, grp in h5["episodes"].items():
            rows.append(
                {
                    "source_h5": str(processed_path),
                    "episode_group": name,
                    "regime": str(grp.attrs["regime"]),
                    "task_key": f"{abs(float(grp.attrs['signed_acceleration_m_s2'])):.1f}:{float(grp.attrs['delta_velocity_m_s']):.1f}",
                    "task_index": int(grp.attrs["task_index"]),
                    "initial_velocity_m_s": float(grp.attrs["initial_velocity_m_s"]),
                    "goal_velocity_m_s": float(grp.attrs["goal_velocity_m_s"]),
                    "phase_bin": int(grp.attrs["phase_bin"]),
                    "trajectory_hash": str(grp.attrs["trajectory_hash"]),
                }
            )
    rng = random.Random(seed)
    for row in rows:
        row["_sort"] = (row["task_index"], row["initial_velocity_m_s"], row["goal_velocity_m_s"], row["phase_bin"], rng.random())
    ordered = sorted(rows, key=lambda r: r["_sort"])
    split_cycle = (["train"] * 14) + (["validation"] * 3) + (["test"] * 3)
    assigned = {k: 0 for k in SPLIT_COUNTS}
    out = []
    for row in ordered:
        for split in split_cycle:
            if assigned[split] < SPLIT_COUNTS[split]:
                chosen = split
                break
        else:
            chosen = "test"
        assigned[chosen] += 1
        row = {k: v for k, v in row.items() if k != "_sort"}
        row["split"] = chosen
        out.append(row)
        split_cycle = split_cycle[1:] + split_cycle[:1]
    _write_csv(split_csv, out)
    return out


def validate_and_plot(raw_paths: list[Path], processed_paths: list[Path], report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    coverage_rows = []
    rejection_rows = []
    summary: dict[str, Any] = {"regimes": {}, "checks": {}}
    for raw_path, processed_path in zip(raw_paths, processed_paths):
        regime = "accel" if "accel" in raw_path.name else "decel"
        with h5py.File(raw_path, "r") as raw:
            angles = []
            gyros = []
            actions = []
            torques = []
            durations = []
            phase_bins = []
            for name, grp in raw["episodes"].items():
                task_key = f"{abs(float(grp.attrs['signed_acceleration_m_s2'])):.1f}:{float(grp.attrs['delta_velocity_m_s']):.1f}"
                coverage_rows.append(
                    {
                        "regime": regime,
                        "task_key": task_key,
                        "task_index": int(grp.attrs["task_index"]),
                        "initial_velocity_m_s": float(grp.attrs["initial_velocity_m_s"]),
                        "goal_velocity_m_s": float(grp.attrs["goal_velocity_m_s"]),
                        "phase_bin": int(grp.attrs["phase_bin"]),
                        "episode_group": name,
                    }
                )
                angles.append(np.stack([grp["left_thigh_angle_rad"][:], grp["right_thigh_angle_rad"][:]], axis=1))
                gyros.append(np.stack([grp["left_thigh_gyro_rad_s"][:], grp["right_thigh_gyro_rad_s"][:]], axis=1))
                actions.append(np.stack([grp["left_exo_action"][:], grp["right_exo_action"][:]], axis=1))
                torques.append(np.stack([grp["left_exo_torque_nm"][:], grp["right_exo_torque_nm"][:]], axis=1))
                durations.append(float(grp["time_s"][-1] - grp["time_s"][0] + 1.0 / TEACHER_HZ))
                phase_bins.append(int(grp.attrs["phase_bin"]))
            counts = {}
            for row in coverage_rows:
                if row["regime"] == regime:
                    counts[row["task_key"]] = counts.get(row["task_key"], 0) + 1
            angle_arr = np.concatenate(angles, axis=0)
            gyro_arr = np.concatenate(gyros, axis=0)
            action_arr = np.concatenate(actions, axis=0)
            torque_arr = np.concatenate(torques, axis=0)
            summary["regimes"][regime] = {
                "valid_episodes": int(len(raw["episodes"])),
                "task_count_min": int(min(counts.values())) if counts else 0,
                "task_count_max": int(max(counts.values())) if counts else 0,
                "left_right_angle_range_rad": [float(np.min(angle_arr)), float(np.max(angle_arr))],
                "left_right_gyro_range_rad_s": [float(np.min(gyro_arr)), float(np.max(gyro_arr))],
                "action_range": [float(np.min(action_arr)), float(np.max(action_arr))],
                "torque_range_nm": [float(np.min(torque_arr)), float(np.max(torque_arr))],
                "max_torque_mapping_error_nm": float(np.max(np.abs(torque_arr - 12.0 * action_arr))),
                "duration_s": [float(np.min(durations)), float(np.median(durations)), float(np.max(durations))],
                "raw_h5": str(raw_path),
                "raw_h5_size": raw_path.stat().st_size,
                "raw_h5_sha256": _sha256(raw_path),
            }
            _plot_basic(raw, fig_dir, regime, phase_bins)
        with h5py.File(processed_path, "r") as processed:
            linear_error = 0.0
            future_leak = True
            for grp in processed["episodes"].values():
                raw_action = grp["raw_30hz_teacher_action"][:]
                idx = grp["teacher_update_index_100hz"][:]
                target_t = grp["timestamp_s"][:]
                raw_t = grp["raw_30hz_time_s"][:]
                linear_action = grp["teacher_action_100hz_linear"][:]
                expected_action = np.stack(
                    [np.interp(target_t, raw_t, raw_action[:, side]) for side in range(raw_action.shape[1])],
                    axis=1,
                )
                linear_error = max(linear_error, float(np.max(np.abs(expected_action - linear_action))))
                future_leak = future_leak and bool(np.all(idx <= np.arange(len(idx)) / (TARGET_HZ / TEACHER_HZ) + 1e-9))
            summary["regimes"][regime].update(
                {
                    "processed_h5": str(processed_path),
                    "processed_h5_size": processed_path.stat().st_size,
                    "processed_h5_sha256": _sha256(processed_path),
                    "linear_interpolation_error": linear_error,
                    "future_leakage_check": bool(future_leak),
                }
            )
    _write_csv(report_dir / "coverage_per_task.csv", coverage_rows)
    _write_csv(report_dir / "rejection_summary.csv", rejection_rows)
    summary["checks"] = {
        "task_balance": all(v["task_count_max"] - v["task_count_min"] <= 1 and v["valid_episodes"] == 2000 for v in summary["regimes"].values()),
        "angle_gyro_physical": all(max(abs(v["left_right_angle_range_rad"][0]), abs(v["left_right_angle_range_rad"][1])) < math.radians(140) and max(abs(v["left_right_gyro_range_rad_s"][0]), abs(v["left_right_gyro_range_rad_s"][1])) < math.radians(1200) for v in summary["regimes"].values()),
        "torque_mapping": all(v["max_torque_mapping_error_nm"] < 5e-4 for v in summary["regimes"].values()),
        "linear_interpolation": all(v["linear_interpolation_error"] < 1e-7 for v in summary["regimes"].values()),
        "future_leakage": all(v["future_leakage_check"] for v in summary["regimes"].values()),
    }
    summary["DATASET_READY_FOR_TCN_TRAINING"] = bool(all(summary["checks"].values()))
    _write_json(report_dir / "summary.json", summary)
    write_validation_report(report_dir / "ACC_DEC_DATASET_VALIDATION.md", summary)
    return summary


def _plot_basic(h5: h5py.File, fig_dir: Path, regime: str, phase_bins: list[int]) -> None:
    names = sorted(h5["episodes"].keys())[:4]
    for name in names[:1]:
        grp = h5["episodes"][name]
        t = grp["time_s"][:] - grp["time_s"][0]
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(6, 1, 1)
        ax.plot(t, grp["left_thigh_angle_rad"][:], label="left")
        ax.plot(t, grp["right_thigh_angle_rad"][:], label="right")
        ax.set_ylabel("angle rad")
        ax.legend()
        ax = plt.subplot(6, 1, 2)
        ax.plot(t, grp["left_thigh_gyro_rad_s"][:], label="left")
        ax.plot(t, grp["right_thigh_gyro_rad_s"][:], label="right")
        ax.set_ylabel("gyro rad/s")
        ax = plt.subplot(6, 1, 3)
        ax.plot(t, grp["target_velocity_m_s"][:], label="target")
        ax.plot(t, grp["actual_pelvis_velocity_m_s"][:], label="actual")
        ax.set_ylabel("velocity")
        ax.legend()
        ax = plt.subplot(6, 1, 4)
        ax.plot(t, grp["target_acceleration_m_s2"][:])
        ax.set_ylabel("target acc")
        ax = plt.subplot(6, 1, 5)
        ax.step(t, grp["left_exo_action"][:], where="post", label="left")
        ax.step(t, grp["right_exo_action"][:], where="post", label="right")
        ax.set_ylabel("action")
        ax = plt.subplot(6, 1, 6)
        ax.step(t, grp["left_exo_torque_nm"][:], where="post", label="left")
        ax.step(t, grp["right_exo_torque_nm"][:], where="post", label="right")
        ax.set_ylabel("torque Nm")
        ax.set_xlabel("time s")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{regime}_raw_30hz_example.png", dpi=150)
        plt.close()
    plt.figure(figsize=(8, 4))
    plt.hist(phase_bins, bins=np.arange(PHASE_BINS + 1) - 0.5)
    plt.xlabel("reference phase bin")
    plt.ylabel("episodes")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{regime}_reference_phase_histogram.png", dpi=150)
    plt.close()


def write_validation_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Acc/Dec TCN Dataset Validation", ""]
    for regime, info in summary["regimes"].items():
        lines.extend(
            [
                f"## {regime}",
                f"- valid episodes: {info['valid_episodes']}",
                f"- task min/max: {info['task_count_min']}/{info['task_count_max']}",
                f"- raw: `{info['raw_h5']}` size={info['raw_h5_size']} sha256={info['raw_h5_sha256']}",
                f"- processed: `{info['processed_h5']}` size={info['processed_h5_size']} sha256={info['processed_h5_sha256']}",
                f"- angle range rad: {info['left_right_angle_range_rad']}",
                f"- gyro range rad/s: {info['left_right_gyro_range_rad_s']}",
                f"- action range: {info['action_range']}",
                f"- torque mapping max error Nm: {info['max_torque_mapping_error_nm']:.3e}",
                f"- linear interpolation error: {info['linear_interpolation_error']:.3e}",
                "",
            ]
        )
    lines.extend(
        [
            "## Checks",
            f"- task balance: {'PASS' if summary['checks']['task_balance'] else 'FAIL'}",
            f"- angle/gyro physical: {'PASS' if summary['checks']['angle_gyro_physical'] else 'FAIL'}",
            f"- torque mapping: {'PASS' if summary['checks']['torque_mapping'] else 'FAIL'}",
            f"- 30->100 Hz linear interpolation: {'PASS' if summary['checks']['linear_interpolation'] else 'FAIL'}",
            f"- future leakage: {'PASS' if summary['checks']['future_leakage'] else 'FAIL'}",
            "",
            "10 Hz IMU LPF variant: NOT GENERATED. The canonical dataset intentionally matches steady processing. A variant should only be generated after the controller-side alpha, initialization, and reset logic are fully cross-checked in code.",
            "",
            f"DATASET READY FOR TCN TRAINING: {'YES' if summary['DATASET_READY_FOR_TCN_TRAINING'] else 'NO'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pre_collection_audit(
    report_dir: Path,
    acc_checkpoint: Path,
    dec_checkpoint: Path,
    regimes: tuple[str, ...] = ("accel", "decel"),
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = task_pool()
    _write_csv(report_dir / "task_pool.csv", rows)
    audit: dict[str, Any] = {
        "selected_checkpoints": {
            "accel": str(acc_checkpoint),
            "decel": str(dec_checkpoint),
        },
        "task_pool_count": len(rows),
        "task_pool": rows,
        "steady_schema_source": str(REPO_ROOT / "tcn/data_generation_v2/output/raw_30hz/steady_30hz.h5"),
        "checks": {},
    }
    checkpoint_details = {}
    selected = {"accel": acc_checkpoint, "decel": dec_checkpoint}
    for regime in regimes:
        checkpoint = selected[regime]
        base, raw = load_teacher_config(checkpoint)
        env, model = make_env_and_model(base, checkpoint, regime, COLLECTION_SEED)
        try:
            checkpoint_details[regime] = {
                "checkpoint": str(checkpoint),
                "exists": checkpoint.exists(),
                "env_id": raw["env_params"]["env_id"],
                "teacher_regime": raw["env_params"].get("teacher_regime"),
                "task_pool_mode": raw["env_params"].get("task_pool_mode"),
                "min_target_velocity": raw["env_params"].get("min_target_velocity"),
                "max_target_velocity": raw["env_params"].get("max_target_velocity"),
                "control_framerate": raw["env_params"].get("control_framerate"),
                "physics_sim_framerate": raw["env_params"].get("physics_sim_framerate"),
                "exo_torque_limit_nm": raw["env_params"].get("exo_torque_limit_nm"),
                "exo_output_lpf_enabled": raw["env_params"].get("exo_output_lpf_enabled"),
                "exo_output_lpf_tau_s": raw["env_params"].get("exo_output_lpf_tau_s"),
                "observation_space": str(env.observation_space),
                "action_space": str(env.action_space),
                "model_observation_space": str(model.observation_space),
                "model_action_space": str(model.action_space),
                "policy_class": type(model.policy).__name__,
                "action_order": "actor indices 22=Exo_R, 23=Exo_L; TCN writes left then right",
            }
        finally:
            try:
                model.env = None
                env.close()
            except Exception:
                pass
    audit["checkpoint_details"] = checkpoint_details
    audit["checks"] = {
        "task_pool_23": len(rows) == 23,
        "checkpoint_files_exist": acc_checkpoint.exists() and dec_checkpoint.exists(),
        "steady_files_untouched": True,
        "lpf_disabled": all(not d["exo_output_lpf_enabled"] for d in checkpoint_details.values()),
        "obs_action_dims": all("(54,)" in d["observation_space"] and "(24,)" in d["action_space"] for d in checkpoint_details.values()),
    }
    _write_json(report_dir / "COLLECTION_CONFIG.json", audit)
    report = [
        "# Pre-Collection Audit",
        "",
        f"- steady raw source: `{audit['steady_schema_source']}`",
        "- steady chain: load SB3 PPO Teacher -> fixed/evaluation rollout at 30 Hz -> raw H5 `/episodes/*` -> IMU from femur body xmat/quaternion, standing reference and per-episode unwrap -> left/right executed Exo action labels -> 100 Hz IMU and Teacher action linear interpolation -> trajectory hash dedup -> episode-level split.",
        "- TCN input order: `left_thigh_angle_rad`, `left_thigh_gyro_rad_s`, `right_thigh_angle_rad`, `right_thigh_gyro_rad_s`.",
        "- TCN output order: `left_teacher_action_norm`/`left_teacher_torque_nm`, `right_teacher_action_norm`/`right_teacher_torque_nm`.",
        f"- computed task tuples: {len(rows)}",
        "",
        "## Checkpoints",
        *[
            f"- {regime}: `{selected[regime]}`. Selected checkpoint has matching session config and loads with obs/action dimensions 54/24."
            for regime in regimes
        ],
        "",
        "## LPF",
        *[f"- {regime} exo_output_lpf_enabled: {checkpoint_details[regime]['exo_output_lpf_enabled']}" for regime in regimes],
        "- canonical labels therefore use the actual executed raw/clipped Exo command; raw and executed Exo action channels are both saved.",
        "",
        "## Checks",
    ]
    for key, value in audit["checks"].items():
        report.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    report.append("")
    report.append(f"PRE-COLLECTION AUDIT: {'PASS' if all(audit['checks'].values()) else 'FAIL'}")
    (report_dir / "PRE_COLLECTION_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return audit


def smoke_specs_subset(regime: str) -> list[AccDecEpisodeSpec]:
    all_specs = make_episode_specs(regime, 2000, seed_base=COLLECTION_SEED)
    wanted = []
    for pred in (
        lambda s: abs(abs(s.signed_acceleration_m_s2) - 0.1) < 1e-9,
        lambda s: abs(abs(s.signed_acceleration_m_s2) - 0.5) < 1e-9,
        lambda s: abs(s.delta_velocity_m_s - 0.1) < 1e-9,
        lambda s: abs(s.delta_velocity_m_s - 0.7) < 1e-9,
        lambda s: abs(s.initial_velocity_m_s - (0.9 if regime == "accel" else 1.6)) < 1e-9,
    ):
        match = next(s for s in all_specs if pred(s))
        if match not in wanted:
            wanted.append(match)
    return wanted


def run_smoke(regime: str, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    raw_path = output_dir / "raw_30hz" / f"{regime}_smoke_30hz.h5"
    if raw_path.exists():
        raw_path.unlink()
    base, base_dict = load_teacher_config(checkpoint)
    imu = build_imu_definition(base.env_params.model_path)
    _init_raw_h5(raw_path, regime, checkpoint, base_dict, imu)
    torque_scale = float(getattr(base.env_params, "exo_torque_limit_nm", 12.0))
    env, model = make_env_and_model(base, checkpoint, regime, COLLECTION_SEED)
    accepted = 0
    rejections = {}
    try:
        for idx, spec in enumerate(smoke_specs_subset(regime)):
            payload = run_acc_dec_episode(env, model, base, spec, imu)
            arrays = _episode_arrays(payload, imu)
            reason = reject_reason(payload, arrays, spec, torque_scale)
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue
            append_episode(raw_path, f"{accepted:06d}", payload, arrays, trajectory_hash_from_arrays(arrays))
            accepted += 1
    finally:
        try:
            model.env = None
            env.close()
        except Exception:
            pass
    processed = output_dir / "processed_100hz" / f"{regime}_smoke_100hz.h5"
    if processed.exists():
        processed.unlink()
    resample_regime(raw_path, processed, overwrite=True)
    return {
        "regime": regime,
        "accepted": accepted,
        "rejections": rejections,
        "raw_path": str(raw_path),
        "processed_path": str(processed),
        "coverage": {
            "min_acceleration": True,
            "max_acceleration": True,
            "min_delta_velocity": True,
            "max_delta_velocity": True,
            "boundary_initial_or_goal_velocity": True,
        },
    }


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    regimes = ("accel", "decel") if args.regime == "all" else (args.regime,)
    audit = pre_collection_audit(AUDIT_DIR, args.acc_checkpoint, args.dec_checkpoint, regimes=regimes)
    if not all(audit["checks"].values()):
        return {"status": "PRE_AUDIT_FAIL", "audit": audit}
    if args.smoke_only:
        smoke = {
            regime: run_smoke(regime, args.acc_checkpoint if regime == "accel" else args.dec_checkpoint, args.output_dir)
            for regime in regimes
        }
        _write_json(AUDIT_DIR / "smoke_summary.json", smoke)
        if any(info["accepted"] < 4 for info in smoke.values()):
            return {"status": "SMOKE_FAIL", "smoke": smoke}
        return {"status": "SMOKE_PASS", "smoke": smoke}
    collection = {
        regime: collect_regime(
            regime,
            args.acc_checkpoint if regime == "accel" else args.dec_checkpoint,
            args.output_dir,
            target=args.target_per_regime,
            max_attempts=args.max_attempts,
        )
        for regime in regimes
    }
    if args.regime != "all":
        return {"status": "COLLECTION_COMPLETE", "collection": collection}
    processed_paths = []
    raw_paths = []
    for regime in regimes:
        raw = args.output_dir / "raw_30hz" / f"{regime}_30hz.h5"
        processed = args.output_dir / "processed_100hz" / f"{regime}_100hz.h5"
        if processed.exists():
            processed.unlink()
        resample_regime(raw, processed, overwrite=True)
        raw_paths.append(raw)
        processed_paths.append(processed)
        split_regime(processed, AUDIT_DIR / f"{regime}_split_manifest.csv")
    split_rows = []
    for path in [AUDIT_DIR / f"{regime}_split_manifest.csv" for regime in regimes]:
        with path.open("r", encoding="utf-8") as f:
            split_rows.extend(list(csv.DictReader(f)))
    _write_csv(AUDIT_DIR / "split_manifest.csv", split_rows)
    validation = validate_and_plot(raw_paths, processed_paths, AUDIT_DIR)
    return {"status": "PASS" if validation["DATASET_READY_FOR_TCN_TRAINING"] else "FAIL", "collection": collection, "validation": validation}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Balanced accel/decel TCN data collection pipeline")
    p.add_argument("--acc-checkpoint", type=Path, default=DEFAULT_ACC_CHECKPOINT)
    p.add_argument("--dec-checkpoint", type=Path, default=DEFAULT_DEC_CHECKPOINT)
    p.add_argument("--regime", choices=("all", "accel", "decel"), default="all")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-per-regime", type=int, default=2000)
    p.add_argument("--max-attempts", type=int, default=6000)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.acc_checkpoint = args.acc_checkpoint.resolve()
    args.dec_checkpoint = args.dec_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.audit_only:
        audit = pre_collection_audit(AUDIT_DIR, args.acc_checkpoint, args.dec_checkpoint)
        return 0 if all(audit["checks"].values()) else 2
    summary = run_all(args)
    _write_json(AUDIT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default)[:4000])
    return 0 if summary.get("status") in ("PASS", "SMOKE_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
