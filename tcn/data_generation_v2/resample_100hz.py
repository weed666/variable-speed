from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import TARGET_HZ, TEACHER_HZ


CONTINUOUS_MAP = {
    "left_thigh_angle_rad": "left_thigh_angle_rad",
    "left_thigh_angular_velocity_rad_s": "left_thigh_gyro_rad_s",
    "right_thigh_angle_rad": "right_thigh_angle_rad",
    "right_thigh_angular_velocity_rad_s": "right_thigh_gyro_rad_s",
    "actual_pelvis_velocity_m_s": "actual_pelvis_velocity_m_s",
    "target_velocity_m_s": "target_velocity_m_s",
    "left_teacher_action_norm": "left_exo_action",
    "right_teacher_action_norm": "right_exo_action",
    "left_teacher_torque_nm": "left_exo_torque_nm",
    "right_teacher_torque_nm": "right_exo_torque_nm",
}
ZOH_MAP = {
    "left_foot_contact": "left_foot_contact",
    "right_foot_contact": "right_foot_contact",
}


def make_raw_relative_time(num_samples: int, hz: int = TEACHER_HZ) -> np.ndarray:
    return np.arange(num_samples, dtype=np.float64) / float(hz)


def make_target_time(duration_s: float = 5.0, hz: int = TARGET_HZ) -> np.ndarray:
    return np.arange(int(round(duration_s * hz)), dtype=np.float64) / float(hz)


def zoh(raw_t: np.ndarray, raw_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(raw_t, target_t, side="right") - 1
    idx = np.clip(idx, 0, len(raw_t) - 1)
    return raw_values[idx]


def relative_raw_time(grp: h5py.Group) -> np.ndarray:
    values = grp["time_s"][:].astype(np.float64)
    return values - values[0]


def trajectory_hash(grp: h5py.Group) -> str:
    h = hashlib.sha256()
    for key in (
        "left_thigh_angle_rad",
        "right_thigh_angle_rad",
        "left_thigh_gyro_rad_s",
        "right_thigh_gyro_rad_s",
        "left_exo_action",
        "right_exo_action",
    ):
        h.update(np.asarray(grp[key][:], dtype=np.float32).tobytes())
    return h.hexdigest()


def resample_raw_to_100hz(raw_path: Path, output_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "r") as src, h5py.File(output_path, "w") as dst:
        dst.attrs["schema_version"] = "tcn_data_generation_v2_processed_100hz_training_schema"
        dst.attrs["source_raw_30hz"] = str(raw_path)
        dst.attrs["target_frequency_hz"] = TARGET_HZ
        dst.attrs["samples_per_episode"] = 500
        dst.attrs["time_axis_strategy"] = "episode-relative 100 Hz grid using real rollout raw timestamps shifted by first sample"
        dst.attrs["imu_resampling"] = "linear interpolation after per-episode angle unwrap"
        dst.attrs["teacher_action_resampling"] = "linear interpolation from 30 Hz teacher command and executed torque"
        dst.attrs["field_schema"] = "TCNDataset canonical fields"
        for key, value in src.attrs.items():
            if key not in dst.attrs:
                dst.attrs[key] = value
        root = dst.create_group("episodes")
        for name, grp in src["episodes"].items():
            if len(grp["time_s"]) != 150:
                raise ValueError(f"{name} has {len(grp['time_s'])} samples; expected 150")
            raw_t = relative_raw_time(grp)
            target_t = make_target_time(float(grp.attrs.get("episode_duration_s", 5.0)))
            out = root.create_group(name)
            for key, value in grp.attrs.items():
                out.attrs[key] = value
            out.attrs["source_raw_group"] = name
            out.attrs["trajectory_hash"] = trajectory_hash(grp)
            out.create_dataset("timestamp_s", data=target_t)
            out.create_dataset("raw_30hz_time_s", data=raw_t)
            out.create_dataset("source_raw_time_s", data=grp["time_s"][:].astype(np.float64))
            out.create_dataset("source_teacher_update_time_s", data=grp["teacher_update_time_s"][:].astype(np.float64))
            out.create_dataset("task_phase", data=np.zeros(len(target_t), dtype=np.int8))
            for dst_key, src_key in CONTINUOUS_MAP.items():
                values = grp[src_key][:].astype(np.float64)
                out.create_dataset(dst_key, data=np.interp(target_t, raw_t, values).astype(np.float32))
            for dst_key, src_key in ZOH_MAP.items():
                values = grp[src_key][:]
                out.create_dataset(dst_key, data=zoh(raw_t, values, target_t))
            action_pair = np.stack([out["left_teacher_action_norm"][:], out["right_teacher_action_norm"][:]], axis=1).astype(np.float32)
            out.create_dataset("teacher_action_100hz_linear", data=action_pair)
            out.create_dataset("raw_30hz_teacher_action", data=np.stack([grp["left_exo_action"][:], grp["right_exo_action"][:]], axis=1))
            out.create_dataset("goal_velocity_m_s", data=np.full(len(target_t), float(grp.attrs["goal_velocity_m_s"]), dtype=np.float32))
            out.create_dataset("initial_velocity_m_s", data=np.full(len(target_t), float(grp.attrs.get("initial_velocity_m_s", grp.attrs["goal_velocity_m_s"])), dtype=np.float32))
            out.create_dataset("target_acceleration_m_s2", data=np.zeros(len(target_t), dtype=np.float32))
            out.create_dataset("episode_id", data=np.asarray([str(grp.attrs.get("episode_id", name)).encode("utf-8")]))
    return {"processed_path": str(output_path), "source_raw_30hz": str(raw_path), "target_hz": TARGET_HZ}
