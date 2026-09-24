from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class ReferenceMotionManager:
    def __init__(
        self,
        *,
        reference_data: dict,
        reference_data_keys: Sequence[str],
        rng,
        flag_random_ref_index: bool = True,
    ):
        if reference_data is None:
            raise ValueError("reference_data is required for ReferenceMotionManager")
        self.reference_data = reference_data
        self.reference_data_keys = list(reference_data_keys)
        if not self.reference_data_keys:
            raise ValueError("reference_data_keys must not be empty")
        if "series_data" not in reference_data or "metadata" not in reference_data:
            raise ValueError("reference_data must contain series_data and metadata")
        self.length = int(reference_data["metadata"]["resampled_data_length"])
        if self.length <= 0:
            raise ValueError("reference_data length must be positive")
        self.rng = rng
        self.flag_random_ref_index = bool(flag_random_ref_index)
        self._validate_keys()

    @staticmethod
    def load_from_path(path: str, *, control_framerate: int | None = None) -> dict:
        ref_path = Path(path)
        if ref_path.suffix == ".npz":
            ref_npz = np.load(ref_path, allow_pickle=True)
            ref_data = {key: ref_npz[key].item() for key in ref_npz.files}
        elif ref_path.suffix == ".json":
            with ref_path.open("r") as f:
                ref_data = json.load(f)
        else:
            raise ValueError("Unsupported reference format. Use .npz or .json.")
        if control_framerate is not None:
            ReferenceMotionManager.resample_in_place(ref_data, control_framerate)
        return ref_data

    @staticmethod
    def resample_in_place(ref_data: dict, control_framerate: int) -> None:
        if "resampled_series_data" in ref_data:
            return
        ref_data["resampled_series_data"] = {}
        for key in ref_data["series_data"].keys():
            original_data = ref_data["series_data"][key]
            original_data_length = len(original_data)
            original_sample_rate = ref_data["metadata"]["sample_rate"]
            original_x = np.linspace(0, original_data_length - 1, original_data_length)
            new_length = int(original_data_length * control_framerate / original_sample_rate)
            new_x = np.linspace(0, original_data_length - 1, new_length)
            ref_data["series_data"][key] = np.interp(new_x, original_x, original_data)
            ref_data["metadata"]["resampled_data_length"] = new_length
            ref_data["metadata"]["resampled_sample_rate"] = control_framerate

    def _validate_keys(self) -> None:
        series_data = self.reference_data["series_data"]
        missing_keys = []
        for key in self.reference_data_keys:
            for prefix in ("q", "dq"):
                ref_key = f"{prefix}_{key}"
                if ref_key not in series_data:
                    missing_keys.append(ref_key)
        if "dq_pelvis_tx" not in series_data:
            missing_keys.append("dq_pelvis_tx")
        if missing_keys:
            raise ValueError(f"reference_data is missing keys: {missing_keys}")

    def select_reset_index(self, fixed_index: int | None = None) -> int:
        if fixed_index is not None:
            index = int(fixed_index)
            if index < 0 or index >= self.length:
                raise ValueError(f"fixed reference index out of range: {index}")
            return index
        if not self.flag_random_ref_index:
            return 0
        high = max(1, int(self.length * 0.8))
        return int(self.rng.integers(0, high))

    def apply_reset_state(
        self,
        sim,
        *,
        reference_index: int,
        initial_velocity: float,
        zero_pelvis_tx: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        series_data = self.reference_data["series_data"]
        ref_pelvis_velocity = float(series_data["dq_pelvis_tx"][reference_index])
        speed_ratio = 1.0
        if np.isfinite(ref_pelvis_velocity) and abs(ref_pelvis_velocity) > 1e-9:
            speed_ratio = float(initial_velocity) / ref_pelvis_velocity

        for key in self.reference_data_keys:
            joint = sim.data.joint(f"{key}")
            joint.qpos = series_data[f"q_{key}"][reference_index]
            if zero_pelvis_tx and key == "pelvis_tx":
                joint.qpos = 0
        for key in self.reference_data_keys:
            joint = sim.data.joint(f"{key}")
            joint.qvel = series_data[f"dq_{key}"][reference_index] * speed_ratio

        sim.data.joint("pelvis_tx").qvel[0] = float(initial_velocity)
        sim.forward()
        return sim.data.qpos.copy(), sim.data.qvel.copy()
