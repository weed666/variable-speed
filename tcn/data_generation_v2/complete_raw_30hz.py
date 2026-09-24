from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .collect_teacher_30hz import EpisodeSpec, load_teacher_config, run_episode, teacher_result_dir
from .config import DEFAULT_OUTPUT, TEACHER_HZ
from .imu_kinematics import build_imu_definition, continuous_angles


REPORT_ROOT = DEFAULT_OUTPUT / "reports/raw_completion"
RAW_KEYS = (
    "time_s",
    "teacher_update_time_s",
    "goal_velocity_m_s",
    "actual_pelvis_velocity_m_s",
    "target_velocity_m_s",
    "left_thigh_xmat",
    "right_thigh_xmat",
    "left_thigh_quat",
    "right_thigh_quat",
    "left_raw_sagittal_angle_rad",
    "right_raw_sagittal_angle_rad",
    "left_thigh_angle_rad",
    "right_thigh_angle_rad",
    "left_thigh_gyro_rad_s",
    "right_thigh_gyro_rad_s",
    "left_world_angular_velocity_rad_s",
    "right_world_angular_velocity_rad_s",
    "left_exo_action",
    "right_exo_action",
    "left_exo_torque_nm",
    "right_exo_torque_nm",
    "left_foot_contact",
    "right_foot_contact",
    "left_hip_angle_rad",
    "right_hip_angle_rad",
    "gait_phase",
    "action_source",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trajectory_hash_from_arrays(left_angle, right_angle, left_gyro, right_gyro, left_action, right_action) -> str:
    h = hashlib.sha256()
    for arr in (left_angle, right_angle, left_gyro, right_gyro, left_action, right_action):
        h.update(np.asarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def trajectory_hash_group(grp: h5py.Group) -> str:
    return trajectory_hash_from_arrays(
        grp["left_thigh_angle_rad"][:],
        grp["right_thigh_angle_rad"][:],
        grp["left_thigh_gyro_rad_s"][:],
        grp["right_thigh_gyro_rad_s"][:],
        grp["left_exo_action"][:],
        grp["right_exo_action"][:],
    )


def is_complete_episode(grp: h5py.Group) -> bool:
    if len(grp["time_s"]) != 150:
        return False
    if str(grp.attrs.get("termination_reason", "")) != "time_limit":
        return False
    try:
        term = json.loads(str(grp.attrs.get("termination_json", "{}")))
    except json.JSONDecodeError:
        return False
    return term.get("reason") == "time_limit" and not bool(term.get("fall", False))


def _copy_root_attrs(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_episode(src_grp: h5py.Group, dst_root: h5py.Group, out_name: str) -> None:
    dst = dst_root.create_group(out_name)
    for key, value in src_grp.attrs.items():
        dst.attrs[key] = value
    for key in src_grp.keys():
        src_grp.copy(key, dst, name=key)


def _write_payload_group(dst_root: h5py.Group, out_name: str, payload: dict[str, Any], imu) -> None:
    rows = payload["rows"]
    spec = payload["spec"]
    grp = dst_root.create_group(out_name)
    grp.attrs["episode_id"] = spec["episode_id"]
    grp.attrs["random_seed"] = int(spec["seed"])
    grp.attrs["goal_velocity_m_s"] = float(spec["velocity_m_s"])
    grp.attrs["episode_duration_s"] = float(spec["duration_s"])
    grp.attrs["reference_index"] = int(spec["reference_index"])
    grp.attrs["regime"] = "steady"
    grp.attrs["initial_velocity_m_s"] = float(spec["velocity_m_s"])
    grp.attrs["target_acceleration_m_s2"] = 0.0
    grp.attrs["signed_acceleration_m_s2"] = 0.0
    grp.attrs["delta_velocity_m_s"] = 0.0
    grp.attrs["task_id"] = "steady"
    grp.attrs["termination_reason"] = payload["termination"]["reason"]
    grp.attrs["termination_json"] = json.dumps(payload["termination"], sort_keys=True)
    raw_left = np.asarray([r["left_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    raw_right = np.asarray([r["right_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    left_angle, right_angle = continuous_angles(raw_left, raw_right, imu)
    left_action = np.asarray([r["left_exo_action"] for r in rows], dtype=np.float32)
    right_action = np.asarray([r["right_exo_action"] for r in rows], dtype=np.float32)
    values: dict[str, Any] = {
        "time_s": np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        "teacher_update_time_s": np.asarray([r["teacher_update_time_s"] for r in rows], dtype=np.float64),
        "teacher_update_index": np.asarray([r["teacher_update_index"] for r in rows], dtype=np.int32),
        "goal_velocity_m_s": np.asarray([r["goal_velocity_m_s"] for r in rows], dtype=np.float32),
        "actual_pelvis_velocity_m_s": np.asarray([r["actual_pelvis_velocity_m_s"] for r in rows], dtype=np.float32),
        "target_velocity_m_s": np.asarray([r["target_velocity_m_s"] for r in rows], dtype=np.float32),
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
        "left_exo_action": left_action,
        "right_exo_action": right_action,
        "left_exo_torque_nm": (12.0 * left_action).astype(np.float32),
        "right_exo_torque_nm": (12.0 * right_action).astype(np.float32),
        "left_foot_contact": np.asarray([r["left_foot_contact"] for r in rows], dtype=np.bool_),
        "right_foot_contact": np.asarray([r["right_foot_contact"] for r in rows], dtype=np.bool_),
        "left_hip_angle_rad": np.asarray([r["left_hip_angle_rad"] for r in rows], dtype=np.float32),
        "right_hip_angle_rad": np.asarray([r["right_hip_angle_rad"] for r in rows], dtype=np.float32),
    }
    for key, value in values.items():
        grp.create_dataset(key, data=value)
    grp.create_dataset("gait_phase", data=np.asarray([str(r["task_phase"]).encode("utf-8") for r in rows]))
    grp.create_dataset("action_source", data=np.asarray([str(r["action_source"]).encode("utf-8") for r in rows]))


def _payload_arrays(payload: dict[str, Any], imu) -> dict[str, np.ndarray]:
    rows = payload["rows"]
    raw_left = np.asarray([r["left_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    raw_right = np.asarray([r["right_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
    left_angle, right_angle = continuous_angles(raw_left, raw_right, imu)
    left_action = np.asarray([r["left_exo_action"] for r in rows], dtype=np.float32)
    right_action = np.asarray([r["right_exo_action"] for r in rows], dtype=np.float32)
    return {
        "time_s": np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        "left_angle": left_angle.astype(np.float32),
        "right_angle": right_angle.astype(np.float32),
        "left_gyro": np.asarray([r["left_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
        "right_gyro": np.asarray([r["right_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
        "left_action": left_action,
        "right_action": right_action,
        "left_torque": (12.0 * left_action).astype(np.float32),
        "right_torque": (12.0 * right_action).astype(np.float32),
    }


def validate_payload(payload: dict[str, Any], imu, existing_hashes: set[str]) -> tuple[bool, str, str | None]:
    rows = payload["rows"]
    if payload["termination"].get("reason") != "time_limit":
        return False, "termination_not_time_limit", None
    if len(rows) != 150:
        return False, f"sample_count_{len(rows)}", None
    arrays = _payload_arrays(payload, imu)
    t = arrays["time_s"]
    if not np.allclose(np.diff(t), 1.0 / TEACHER_HZ, atol=1e-9):
        return False, "timestamp_not_30hz", None
    for key, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            return False, f"nonfinite_{key}", None
    if np.max(np.abs(np.diff(arrays["left_angle"]))) > np.deg2rad(90.0):
        return False, "left_branch_flip", None
    if np.max(np.abs(np.diff(arrays["right_angle"]))) > np.deg2rad(90.0):
        return False, "right_branch_flip", None
    if np.max(np.abs(arrays["left_action"])) > 1.0 + 1e-6 or np.max(np.abs(arrays["right_action"])) > 1.0 + 1e-6:
        return False, "action_out_of_range", None
    if np.max(np.abs(arrays["left_torque"] - 12.0 * arrays["left_action"])) > 1e-6:
        return False, "left_torque_scale", None
    if np.max(np.abs(arrays["right_torque"] - 12.0 * arrays["right_action"])) > 1e-6:
        return False, "right_torque_scale", None
    h = trajectory_hash_from_arrays(
        arrays["left_angle"],
        arrays["right_angle"],
        arrays["left_gyro"],
        arrays["right_gyro"],
        arrays["left_action"],
        arrays["right_action"],
    )
    if h in existing_hashes:
        return False, "duplicate_trajectory_hash", h
    return True, "accepted", h


def complete_raw(raw_path: Path, checkpoint: Path, *, seed_base: int = 191815936, max_attempts_per_episode: int = 200) -> dict[str, Any]:
    report_root = REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    backup_path = raw_path.with_name("steady_30hz.before_completion.h5")
    tmp_path = raw_path.with_name("steady_30hz.completed.tmp.h5")
    if not backup_path.exists():
        shutil.copy2(raw_path, backup_path)
    backup_sha = sha256_file(backup_path)
    original_sha = sha256_file(raw_path)
    if backup_sha != original_sha:
        raise RuntimeError("existing backup SHA differs from active raw file; refusing to continue")

    base_config, _base_config_dict = load_teacher_config(checkpoint)
    imu = build_imu_definition(base_config.env_params.model_path)
    missing: dict[float, int] = {}
    existing_hashes_by_speed: dict[float, set[str]] = {}
    good_groups: list[str] = []
    bad_rows: list[dict[str, Any]] = []
    with h5py.File(raw_path, "r") as src:
        for name, grp in src["episodes"].items():
            v = round(float(grp.attrs["goal_velocity_m_s"]), 2)
            existing_hashes_by_speed.setdefault(v, set())
            if is_complete_episode(grp):
                good_groups.append(name)
                existing_hashes_by_speed[v].add(trajectory_hash_group(grp))
            else:
                missing[v] = missing.get(v, 0) + 1
                bad_rows.append(
                    {
                        "source_group": name,
                        "goal_velocity_m_s": v,
                        "sample_count": len(grp["time_s"]),
                        "termination_reason": grp.attrs.get("termination_reason", ""),
                    }
                )
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(tmp_path, "w") as dst:
            _copy_root_attrs(src, dst)
            dst.attrs["completion_backup_sha256"] = backup_sha
            dst.attrs["completion_source_raw"] = str(raw_path)
            root = dst.create_group("episodes")
            out_idx = 0
            for name in sorted(good_groups):
                _copy_episode(src["episodes"][name], root, f"{out_idx:06d}")
                out_idx += 1

            accepted_rows: list[dict[str, Any]] = []
            rejected_rows: list[dict[str, Any]] = []
            for v in sorted(missing):
                needed = missing[v]
                accepted = 0
                attempts = 0
                used_refs = {
                    int(src["episodes"][name].attrs.get("reference_index", -1))
                    for name in good_groups
                    if round(float(src["episodes"][name].attrs["goal_velocity_m_s"]), 2) == v
                }
                while accepted < needed:
                    if attempts >= max_attempts_per_episode * needed:
                        raise RuntimeError(f"failed to collect {needed} replacements for {v:.2f} m/s")
                    seed = int(seed_base + round(v * 100) * 100_000 + attempts)
                    ref_candidates = [idx for idx in range(90) if idx not in used_refs]
                    ref = ref_candidates[attempts % len(ref_candidates)] if ref_candidates else attempts % 90
                    spec = EpisodeSpec(
                        episode_id=f"steady_v{int(round(v * 100)):03d}_completion_{accepted:04d}",
                        velocity_m_s=float(v),
                        seed=seed,
                        reference_index=int(ref),
                        duration_s=5.0,
                    )
                    print(f"[complete30] v={v:.2f} accepted={accepted}/{needed} attempt={attempts} seed={seed} ref={ref}", flush=True)
                    payload = run_episode(base_config, checkpoint, spec, imu)
                    ok, reason, traj_hash = validate_payload(payload, imu, existing_hashes_by_speed[v])
                    attempts += 1
                    if not ok:
                        rejected_rows.append(
                            {
                                "goal_velocity_m_s": v,
                                "seed": seed,
                                "reference_index": ref,
                                "reason": reason,
                                "sample_count": len(payload["rows"]),
                                "termination_reason": payload["termination"].get("reason"),
                            }
                        )
                        continue
                    _write_payload_group(root, f"{out_idx:06d}", payload, imu)
                    existing_hashes_by_speed[v].add(str(traj_hash))
                    accepted_rows.append(
                        {
                            "episode_group": f"{out_idx:06d}",
                            "goal_velocity_m_s": v,
                            "seed": seed,
                            "reference_index": ref,
                            "trajectory_hash": traj_hash,
                        }
                    )
                    out_idx += 1
                    accepted += 1

    _write_csv(report_root / "excluded_episodes.csv", bad_rows)
    _write_csv(report_root / "accepted_replacements.csv", accepted_rows)
    _write_csv(report_root / "rejected_candidates.csv", rejected_rows)
    summary = {
        "raw_path": str(raw_path),
        "backup_path": str(backup_path),
        "tmp_path": str(tmp_path),
        "backup_sha256": backup_sha,
        "original_sha256": original_sha,
        "kept_complete_episodes": len(good_groups),
        "missing_by_speed": {f"{k:.2f}": v for k, v in sorted(missing.items())},
        "accepted_replacements": len(accepted_rows),
        "rejected_candidates": len(rejected_rows),
    }
    (report_root / "completion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-base", type=int, default=191815936)
    args = p.parse_args(argv)
    summary = complete_raw(args.raw_path.resolve(), args.checkpoint.resolve(), seed_base=args.seed_base)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
