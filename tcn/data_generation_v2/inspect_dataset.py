from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcn-data-generation-v2-final-mpl")
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DEFAULT_OUTPUT, HISTORY_STEPS, INPUT_CHANNELS, TARGET_CHANNELS


REPORT_ROOT = DEFAULT_OUTPUT / "audits/final"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flip_count(x: np.ndarray) -> int:
    return int(np.sum(np.abs(np.diff(x)) > math.radians(90.0)))


def final_audit(dataset_dir: Path, report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    fig_dir = report_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    norm = json.loads((dataset_dir / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(norm["input_mean"], dtype=np.float64)
    std = np.asarray(norm["input_std"], dtype=np.float64)
    stats_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    all_episode_ids: dict[str, set[str]] = {}
    total_flips = 0
    linear_pass = True
    dt_pass = True
    finite_pass = True
    speeds_seen = set()
    per_velocity_counts: dict[float, int] = {}
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.h5"
        ids = set()
        with h5py.File(path, "r") as h5:
            for ep_name, grp in h5["episodes"].items():
                t = grp["timestamp_s"][:]
                dt_pass &= bool(np.allclose(np.diff(t), 0.01, atol=1e-9))
                ep_id = str(grp.attrs.get("episode_id", ep_name))
                ids.add(ep_id)
                v = float(grp.attrs["goal_velocity_m_s"])
                speeds_seen.add(round(v, 2))
                per_velocity_counts[v] = per_velocity_counts.get(v, 0) + 1
                x = np.stack([grp[ch][:] for ch in INPUT_CHANNELS], axis=0).astype(np.float64)
                y = np.stack([grp[ch][:] for ch in TARGET_CHANNELS], axis=1).astype(np.float64)
                finite_pass &= bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
                lf = _flip_count(x[0])
                rf = _flip_count(x[2])
                total_flips += lf + rf
                raw_t = grp["raw_30hz_time_s"][:]
                raw_action = grp["raw_30hz_teacher_action"][:]
                action = grp["teacher_action_100hz_linear"][:]
                expected_action = np.stack(
                    [np.interp(t, raw_t, raw_action[:, side]) for side in range(raw_action.shape[1])],
                    axis=1,
                )
                linear_pass &= bool(np.allclose(action, expected_action, atol=1e-7))
                for cidx, ch in enumerate(INPUT_CHANNELS):
                    stats_rows.append(
                        {
                            "split": split,
                            "episode": ep_name,
                            "episode_id": ep_id,
                            "goal_velocity_m_s": v,
                            "channel_index": cidx,
                            "channel": ch,
                            "min": float(np.min(x[cidx])),
                            "max": float(np.max(x[cidx])),
                            "mean": float(np.mean(x[cidx])),
                            "std": float(np.std(x[cidx])),
                            "branch_flips": lf if cidx == 0 else rf if cidx == 2 else 0,
                        }
                    )
                if split == "train" and not example_rows and x.shape[1] >= HISTORY_STEPS:
                    raw_window = x[:, :HISTORY_STEPS]
                    norm_window = (raw_window - mean[:, None]) / std[:, None]
                    inv = norm_window * std[:, None] + mean[:, None]
                    for i in range(HISTORY_STEPS):
                        row = {"sample": i, "timestamp_s": float(t[i])}
                        for cidx, ch in enumerate(INPUT_CHANNELS):
                            row[f"{ch}_physical"] = float(inv[cidx, i])
                            row[f"{ch}_normalized"] = float(norm_window[cidx, i])
                        row["left_exo_action"] = float(y[i, 0])
                        row["right_exo_action"] = float(y[i, 1])
                        example_rows.append(row)
        all_episode_ids[split] = ids
        split_rows.append({"split": split, "episodes": len(ids)})
    leakage_pass = not (all_episode_ids["train"] & all_episode_ids["validation"] or all_episode_ids["train"] & all_episode_ids["test"] or all_episode_ids["validation"] & all_episode_ids["test"])
    expected_speeds = {v / 100.0 for v in range(90, 161, 5)}
    speed_pass = speeds_seen == expected_speeds
    _write_csv(report_root / "dataset_statistics.csv", stats_rows)
    _write_csv(report_root / "split_statistics.csv", split_rows)
    _write_csv(report_root / "final_tcn_input_examples.csv", example_rows)
    if example_rows:
        t = [r["timestamp_s"] for r in example_rows]
        plt.figure(figsize=(10, 5))
        plt.plot(t, [np.degrees(r["left_thigh_angle_rad_physical"]) for r in example_rows], label="left angle")
        plt.plot(t, [np.degrees(r["right_thigh_angle_rad_physical"]) for r in example_rows], label="right angle")
        plt.legend()
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.tight_layout()
        plt.savefig(fig_dir / "final_tcn_input_angle_window.png", dpi=150)
        plt.close()
    summary = {
        "dataset_dir": str(dataset_dir),
        "input_shape": [4, HISTORY_STEPS],
        "input_channel_order": list(INPUT_CHANNELS),
        "target_channel_order": list(TARGET_CHANNELS),
        "angle_branch_flips": total_flips,
        "ANGLE_CONTINUITY": "PASS" if total_flips == 0 else "FAIL",
        "IMU_RESAMPLING": "PASS" if dt_pass and finite_pass else "FAIL",
        "TEACHER_ACTION_LINEAR": "PASS" if linear_pass else "FAIL",
        "TRAIN_VAL_TEST_LEAKAGE": "PASS" if leakage_pass else "FAIL",
        "SPEED_COVERAGE": "PASS" if speed_pass else "FAIL",
        "velocity_counts": {f"{k:.2f}": v for k, v in sorted(per_velocity_counts.items())},
    }
    summary["OVERALL"] = "PASS" if all(summary[k] == "PASS" for k in ["ANGLE_CONTINUITY", "IMU_RESAMPLING", "TEACHER_ACTION_LINEAR", "TRAIN_VAL_TEST_LEAKAGE", "SPEED_COVERAGE"]) else "FAIL"
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# TCN Dataset V2 Audit",
        "",
        f"- OVERALL: {summary['OVERALL']}",
        f"- ANGLE CONTINUITY: {summary['ANGLE_CONTINUITY']}",
        f"- IMU RESAMPLING: {summary['IMU_RESAMPLING']}",
        f"- TEACHER ACTION LINEAR INTERPOLATION: {summary['TEACHER_ACTION_LINEAR']}",
        f"- TRAIN/VAL/TEST LEAKAGE: {summary['TRAIN_VAL_TEST_LEAKAGE']}",
        f"- SPEED COVERAGE: {summary['SPEED_COVERAGE']}",
        f"- angle branch flips: {total_flips}",
        f"- input shape: {summary['input_shape']}",
        f"- channel order: {summary['input_channel_order']}",
    ]
    (report_root / "TCN_DATASET_V2_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
