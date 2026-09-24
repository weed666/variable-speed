from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "tcn" / "dataset_generation" / "output" / "tcn_dataset"

REGIMES = ("steady", "acceleration", "deceleration")
MODE_TO_REGIMES = {
    "steady": ("steady",),
    "acceleration": ("acceleration",),
    "deceleration": ("deceleration",),
    "unified": REGIMES,
}
SPLITS = ("train", "val", "test")
REGIME_TO_ID = {"steady": 0, "acceleration": 1, "deceleration": 2}
ID_TO_REGIME = {v: k for k, v in REGIME_TO_ID.items()}
INPUT_CHANNELS = (
    "left_thigh_angle_rad",
    "left_thigh_angular_velocity_rad_s",
    "right_thigh_angle_rad",
    "right_thigh_angular_velocity_rad_s",
)
TARGET_CHANNELS = ("left_teacher_action_norm", "right_teacher_action_norm")
TORQUE_CHANNELS = ("left_teacher_torque_nm", "right_teacher_torque_nm")


@dataclass(frozen=True)
class EpisodeInfo:
    regime: str
    regime_id: int
    h5_path: Path
    h5_group: str
    episode_id: str
    global_episode_uid: str
    length: int
    valid_windows: int
    task_id: str
    target_acceleration: float
    initial_velocity: float
    goal_velocity: float


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_h5_path(value: str, dataset_root: Path = DATASET_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    return dataset_root / "raw" / path.name


def load_normalization(
    mode: str,
    unified_normalization: str = "regime_balanced",
    normalization_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    stats_path = Path(normalization_path) if normalization_path is not None else DATASET_ROOT / "metadata" / "normalization_stats.json"
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    stats = _load_json(stats_path)
    if not stats.get("train_only", False):
        raise ValueError("normalization_stats.json is not marked train_only=true")
    if tuple(stats["input_channel_order"]) != INPUT_CHANNELS:
        raise ValueError("normalization input channel order does not match TCNDataset input order")

    if mode in REGIMES:
        scheme = mode
        block = stats["per_regime_train"][mode]
    elif mode == "unified":
        if unified_normalization == "regime_balanced":
            scheme = "unified_train_regime_balanced"
        elif unified_normalization == "raw_frame_weighted":
            scheme = "unified_train_raw_frame_weighted"
        else:
            raise ValueError("unified_normalization must be regime_balanced or raw_frame_weighted")
        block = stats[scheme]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    mean = np.asarray(block["input_mean"], dtype=np.float32)
    std = np.asarray(block["input_std"], dtype=np.float32)
    if mean.shape != (len(INPUT_CHANNELS),) or std.shape != (len(INPUT_CHANNELS),):
        raise ValueError(f"Invalid normalization shape for {scheme}")
    return mean, std, scheme


class TCNDataset(Dataset):
    """Lazy HDF5-backed causal TCN window dataset."""

    def __init__(
        self,
        mode: str,
        split: str,
        history_steps: int = 100,
        normalize: bool = True,
        unified_normalization: str = "regime_balanced",
        dataset_root: str | Path | None = None,
        normalization_path: str | Path | None = None,
        include_debug_metadata: bool = False,
    ) -> None:
        if mode not in MODE_TO_REGIMES:
            raise ValueError(f"mode must be one of {tuple(MODE_TO_REGIMES)}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if history_steps != 100:
            raise ValueError("history_steps baseline is fixed at 100 for this loader")

        self.mode = mode
        self.split = split
        self.history_steps = history_steps
        self.normalize = normalize
        self.unified_normalization = unified_normalization
        self.dataset_root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
        if not self.dataset_root.is_absolute():
            self.dataset_root = REPO_ROOT / self.dataset_root
        self.normalization_path = normalization_path if normalization_path is not None else self.dataset_root / "metadata" / "normalization_stats.json"
        self.include_debug_metadata = include_debug_metadata

        self.input_channels = INPUT_CHANNELS
        self.target_channels = TARGET_CHANNELS
        self.mean, self.std, self.normalization_scheme = load_normalization(mode, unified_normalization, self.normalization_path)
        self.std_safe = np.maximum(self.std, np.float32(1e-8)).astype(np.float32)

        self.episode_infos: list[EpisodeInfo] = []
        self.cumulative_windows: list[int] = [0]
        self.regime_index_ranges: dict[int, tuple[int, int]] = {}
        self._h5_handles: dict[Path, h5py.File] = {}
        self._build_index()

    def _build_index(self) -> None:
        total = 0
        for regime in MODE_TO_REGIMES[self.mode]:
            start = total
            split_data = _load_json(self.dataset_root / "splits" / f"{regime}_split.json")
            rows = split_data["episodes"][self.split]
            h5_paths = sorted({_resolve_h5_path(row["h5_file"], self.dataset_root) for row in rows})
            lengths: dict[str, int] = {}
            for path in h5_paths:
                with h5py.File(path, "r") as h5:
                    for row in rows:
                        row_path = _resolve_h5_path(row["h5_file"], self.dataset_root)
                        if row_path != path:
                            continue
                        group_name = row["h5_group"].strip("/")
                        lengths[group_name] = int(len(h5[group_name][INPUT_CHANNELS[0]]))

            for row in rows:
                group_name = row["h5_group"].strip("/")
                length = lengths[group_name]
                valid_windows = length - self.history_steps + 1
                if valid_windows <= 0:
                    raise ValueError(f"Episode {row['global_episode_uid']} has no valid 100-step windows")
                info = EpisodeInfo(
                    regime=regime,
                    regime_id=REGIME_TO_ID[regime],
                    h5_path=_resolve_h5_path(row["h5_file"], self.dataset_root),
                    h5_group=group_name,
                    episode_id=str(row["episode_id"]),
                    global_episode_uid=str(row["global_episode_uid"]),
                    length=length,
                    valid_windows=valid_windows,
                    task_id=str(row.get("task_id", "")),
                    target_acceleration=float(row.get("target_acceleration", 0.0)),
                    initial_velocity=float(row.get("initial_velocity", 0.0)),
                    goal_velocity=float(row.get("goal_velocity", 0.0)),
                )
                self.episode_infos.append(info)
                total += valid_windows
                self.cumulative_windows.append(total)
            self.regime_index_ranges[REGIME_TO_ID[regime]] = (start, total)

    def __len__(self) -> int:
        return self.cumulative_windows[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def _get_handle(self, path: Path) -> h5py.File:
        handle = self._h5_handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._h5_handles[path] = handle
        return handle

    def close(self) -> None:
        for handle in list(self._h5_handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles.clear()

    def __del__(self) -> None:
        self.close()

    def index_to_episode_endpoint(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_idx = bisect.bisect_right(self.cumulative_windows, index) - 1
        within_episode = index - self.cumulative_windows[episode_idx]
        endpoint = within_episode + self.history_steps - 1
        return episode_idx, endpoint

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_idx, endpoint = self.index_to_episode_endpoint(index)
        info = self.episode_infos[episode_idx]
        h5 = self._get_handle(info.h5_path)
        group = h5[info.h5_group]
        start = endpoint - self.history_steps + 1
        stop = endpoint + 1

        raw = np.stack([group[ch][start:stop] for ch in INPUT_CHANNELS], axis=0).astype(np.float32)
        if self.normalize:
            x = (raw - self.mean[:, None]) / self.std_safe[:, None]
        else:
            x = raw
        y = np.asarray([group[ch][endpoint] for ch in TARGET_CHANNELS], dtype=np.float32)

        item: dict[str, Any] = {
            "x": torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False))),
            "y": torch.from_numpy(y),
            "regime": info.regime_id,
            "episode_id": info.episode_id,
            "global_episode_uid": info.global_episode_uid,
            "endpoint_index": int(endpoint),
        }
        if self.include_debug_metadata:
            item.update(
                {
                    "timestamp": float(group["timestamp_s"][endpoint]),
                    "task_phase": int(group["task_phase"][endpoint]),
                    "teacher_torque_nm": torch.tensor(
                        [group[ch][endpoint] for ch in TORQUE_CHANNELS], dtype=torch.float32
                    ),
                    "task_id": info.task_id,
                    "target_acceleration": info.target_acceleration,
                    "initial_velocity": info.initial_velocity,
                    "goal_velocity": info.goal_velocity,
                }
            )
        return item

    def window_counts_by_regime(self) -> dict[str, int]:
        counts = {regime: 0 for regime in REGIMES}
        for info in self.episode_infos:
            counts[info.regime] += info.valid_windows
        if self.mode != "unified":
            return {self.mode: counts[self.mode]}
        return counts

    def episode_ids_by_regime(self) -> dict[str, set[str]]:
        out = {regime: set() for regime in REGIMES}
        for info in self.episode_infos:
            out[info.regime].add(info.global_episode_uid)
        return out

    def indexing_metadata_bytes_estimate(self) -> int:
        return (
            len(self.episode_infos) * 256
            + len(self.cumulative_windows) * 8
            + len(self.regime_index_ranges) * 32
        )

    def worker_pid(self) -> int:
        return os.getpid()
