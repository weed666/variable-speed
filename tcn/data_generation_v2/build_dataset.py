from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import HISTORY_STEPS, INPUT_CHANNELS, TARGET_CHANNELS


SPLITS = ("train", "validation", "test")


def _split_names_by_velocity(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((float(row["goal_velocity_m_s"]), str(row["trajectory_hash"])), []).append(row)
    by_v: dict[float, list[list[dict[str, Any]]]] = {}
    for (velocity, _hash), group in grouped.items():
        by_v.setdefault(velocity, []).append(group)
    out = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for _v, groups in sorted(by_v.items()):
        groups = list(groups)
        rng.shuffle(groups)
        n = sum(len(group) for group in groups)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        train_count = 0
        val_count = 0
        for group in groups:
            if train_count + len(group) <= n_train:
                out["train"].extend(group)
                train_count += len(group)
            elif val_count + len(group) <= n_val:
                out["validation"].extend(group)
                val_count += len(group)
            else:
                out["test"].extend(group)
    return out


def _copy_episode(src_grp: h5py.Group, dst_root: h5py.Group, out_name: str) -> dict[str, Any]:
    out = dst_root.create_group(out_name)
    for key, value in src_grp.attrs.items():
        out.attrs[key] = value
    for key in (
        "timestamp_s",
        "raw_30hz_time_s",
        "raw_30hz_teacher_action",
        *INPUT_CHANNELS,
        *TARGET_CHANNELS,
        "left_teacher_torque_nm",
        "right_teacher_torque_nm",
        "teacher_action_100hz_linear",
        "goal_velocity_m_s",
    ):
        src_key = key
        if key == "left_thigh_gyro_rad_s":
            src_key = "left_thigh_gyro_rad_s"
        if key == "right_thigh_gyro_rad_s":
            src_key = "right_thigh_gyro_rad_s"
        out.create_dataset(key, data=src_grp[src_key][:])
    out.attrs["num_samples"] = len(out["timestamp_s"])
    return {
        "episode_group": out_name,
        "episode_id": str(src_grp.attrs.get("episode_id", out_name)),
        "goal_velocity_m_s": float(src_grp.attrs["goal_velocity_m_s"]),
        "trajectory_hash": str(src_grp.attrs.get("trajectory_hash", "")),
        "num_samples": int(len(out["timestamp_s"])),
        "valid_windows": int(max(0, len(out["timestamp_s"]) - HISTORY_STEPS + 1)),
    }


def build_dataset(processed_path: Path, dataset_dir: Path, *, seed: int, overwrite: bool = False) -> dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    outputs = [dataset_dir / f"{s}.h5" for s in SPLITS]
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    with h5py.File(processed_path, "r") as src:
        rows = [
            {
                "name": name,
                "goal_velocity_m_s": float(grp.attrs["goal_velocity_m_s"]),
                "trajectory_hash": str(grp.attrs["trajectory_hash"]),
            }
            for name, grp in src["episodes"].items()
        ]
        split_rows = _split_names_by_velocity(rows, seed)
        manifest: dict[str, Any] = {
            "schema_version": "tcn_data_generation_v2_dataset_manifest",
            "source_processed_100hz": str(processed_path),
            "history_steps": HISTORY_STEPS,
            "input_channel_order": list(INPUT_CHANNELS),
            "target_channel_order": list(TARGET_CHANNELS),
            "label_definition": "30 Hz teacher normalized exoskeleton action linearly interpolated to 100 Hz",
            "label_units": "normalized action in [-1, 1]; exoskeleton torque command is 12 * action",
            "split_group_key": "goal_velocity_m_s + trajectory_hash",
            "splits": {},
        }
        train_values = []
        for split, items in split_rows.items():
            out_path = dataset_dir / f"{split}.h5"
            with h5py.File(out_path, "w") as dst:
                dst.attrs["schema_version"] = "tcn_data_generation_v2_dataset_split"
                dst.attrs["split"] = split
                dst.attrs["source_processed_100hz"] = str(processed_path)
                dst.attrs["history_steps"] = HISTORY_STEPS
                dst.attrs["input_channel_order_json"] = json.dumps(list(INPUT_CHANNELS))
                dst.attrs["target_channel_order_json"] = json.dumps(list(TARGET_CHANNELS))
                root = dst.create_group("episodes")
                ep_rows = []
                for out_idx, row in enumerate(sorted(items, key=lambda r: r["name"])):
                    copied = _copy_episode(src["episodes"][row["name"]], root, f"{out_idx:06d}")
                    copied["source_group"] = row["name"]
                    ep_rows.append(copied)
                    if split == "train":
                        g = root[f"{out_idx:06d}"]
                        train_values.append(np.stack([g[ch][:] for ch in INPUT_CHANNELS], axis=1).astype(np.float64))
                manifest["splits"][split] = {"path": str(out_path), "episodes": ep_rows}
        train = np.concatenate(train_values, axis=0)
        mean = train.mean(axis=0)
        std = np.maximum(train.std(axis=0), 1e-8)
        norm = {
            "train_only": True,
            "input_channel_order": list(INPUT_CHANNELS),
            "input_mean": mean.tolist(),
            "input_std": std.tolist(),
            "units": {"angle": "rad", "gyro": "rad/s"},
        }
        (dataset_dir / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
        (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"dataset_dir": str(dataset_dir), "normalization": str(dataset_dir / "normalization.json")}
