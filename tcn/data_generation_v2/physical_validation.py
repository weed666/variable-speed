from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcn-data-generation-v2-mpl")
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DEFAULT_OUTPUT


REPORT_ROOT = DEFAULT_OUTPUT / "audits/validation_30hz"
FLIP_THRESHOLD_RAD = math.radians(90.0)


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


def _lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 2) -> tuple[float, int]:
    best = (-np.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[: lag or None]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        if aa.size < 3 or np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
            corr = -np.inf
        else:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        if corr > best[0]:
            best = (corr, lag)
    return float(best[0]), int(best[1])


def _gyro_metrics(t: np.ndarray, angle: np.ndarray, gyro: np.ndarray) -> dict[str, Any]:
    dangle = np.gradient(angle, t)
    corr, lag = _lagged_corr(dangle, gyro)
    if lag < 0:
        aa, gg = dangle[-lag:], gyro[: lag or None]
    elif lag > 0:
        aa, gg = dangle[:-lag], gyro[lag:]
    else:
        aa, gg = dangle, gyro
    slope = float(np.linalg.lstsq(gg.reshape(-1, 1), aa, rcond=None)[0][0]) if np.any(np.abs(gg) > 1e-12) else math.nan
    rmse = float(np.sqrt(np.mean((aa - gg) ** 2))) if aa.size else math.nan
    return {
        "correlation": corr,
        "best_lag_samples": lag,
        "slope_dangle_dt_over_gyro": slope,
        "rmse_rad_s": rmse,
        "sign_consistent": bool(corr > 0.0 and slope > 0.0),
    }


def _branch_events(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.abs(np.diff(values)) > FLIP_THRESHOLD_RAD) + 1


def _contact_period(t: np.ndarray, contact: np.ndarray) -> float:
    contact = np.asarray(contact, dtype=bool)
    onsets = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    if onsets.size < 2:
        return math.nan
    return float(np.median(np.diff(t[onsets])))


def validate_raw_30hz(raw_path: Path, report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    by_velocity: dict[float, list[dict[str, Any]]] = {}

    with h5py.File(raw_path, "r") as h5:
        for ep_name, grp in h5["episodes"].items():
            t = grp["time_s"][:].astype(np.float64)
            dt = np.diff(t)
            goal = float(grp.attrs["goal_velocity_m_s"])
            left_angle = grp["left_thigh_angle_rad"][:].astype(np.float64)
            right_angle = grp["right_thigh_angle_rad"][:].astype(np.float64)
            left_gyro = grp["left_thigh_gyro_rad_s"][:].astype(np.float64)
            right_gyro = grp["right_thigh_gyro_rad_s"][:].astype(np.float64)
            left_action = grp["left_exo_action"][:].astype(np.float64)
            right_action = grp["right_exo_action"][:].astype(np.float64)
            vel = grp["actual_pelvis_velocity_m_s"][:].astype(np.float64)
            lcontact = grp["left_foot_contact"][:].astype(bool)
            rcontact = grp["right_foot_contact"][:].astype(bool)
            lflips = _branch_events(left_angle)
            rflips = _branch_events(right_angle)
            for side, flips, angle in [("left", lflips, left_angle), ("right", rflips, right_angle)]:
                for idx in flips:
                    branch_rows.append(
                        {
                            "episode": ep_name,
                            "side": side,
                            "sample_index": int(idx),
                            "time_s": float(t[idx]),
                            "jump_rad": float(angle[idx] - angle[idx - 1]),
                            "jump_deg": float(np.degrees(angle[idx] - angle[idx - 1])),
                        }
                    )
            lgyro = _gyro_metrics(t, left_angle, left_gyro)
            rgyro = _gyro_metrics(t, right_angle, right_gyro)
            finite = all(
                np.all(np.isfinite(x))
                for x in [t, left_angle, right_angle, left_gyro, right_gyro, left_action, right_action, vel]
            )
            row = {
                "episode": ep_name,
                "episode_id": grp.attrs.get("episode_id", ep_name),
                "goal_velocity_m_s": goal,
                "expected_sample_count": int(round(float(grp.attrs.get("episode_duration_s", 0.0)) * 30.0)),
                "sample_count": int(len(t)),
                "timestamp_strict": bool(np.all(dt > 0)),
                "mean_hz": float(1.0 / np.median(dt)) if dt.size else math.nan,
                "max_dt_error_s": float(np.max(np.abs(dt - 1.0 / 30.0))) if dt.size else math.nan,
                "finite": bool(finite),
                "left_branch_flips": int(len(lflips)),
                "right_branch_flips": int(len(rflips)),
                "left_angle_rom_deg": float(np.degrees(np.max(left_angle) - np.min(left_angle))),
                "right_angle_rom_deg": float(np.degrees(np.max(right_angle) - np.min(right_angle))),
                "left_gyro_peak_deg_s": float(np.degrees(np.max(np.abs(left_gyro)))),
                "right_gyro_peak_deg_s": float(np.degrees(np.max(np.abs(right_gyro)))),
                "left_gyro_corr": lgyro["correlation"],
                "right_gyro_corr": rgyro["correlation"],
                "left_gyro_slope": lgyro["slope_dangle_dt_over_gyro"],
                "right_gyro_slope": rgyro["slope_dangle_dt_over_gyro"],
                "left_gyro_rmse_rad_s": lgyro["rmse_rad_s"],
                "right_gyro_rmse_rad_s": rgyro["rmse_rad_s"],
                "left_contact_period_s": _contact_period(t, lcontact),
                "right_contact_period_s": _contact_period(t, rcontact),
                "velocity_mean_m_s": float(np.nanmean(vel)),
                "velocity_error_mean_m_s": float(np.nanmean(vel - goal)),
                "left_action_min": float(np.nanmin(left_action)),
                "left_action_max": float(np.nanmax(left_action)),
                "right_action_min": float(np.nanmin(right_action)),
                "right_action_max": float(np.nanmax(right_action)),
            }
            episode_rows.append(row)
            by_velocity.setdefault(goal, []).append(row)

    angle_pass = len(branch_rows) == 0 and all(max(r["left_branch_flips"], r["right_branch_flips"]) == 0 for r in episode_rows)
    gyro_pass = all(
        r["left_gyro_corr"] > 0.8
        and r["right_gyro_corr"] > 0.8
        and r["left_gyro_slope"] > 0.2
        and r["right_gyro_slope"] > 0.2
        for r in episode_rows
    )
    gait_pass = all(
        np.isfinite(r["left_contact_period_s"]) or np.isfinite(r["right_contact_period_s"]) for r in episode_rows
    )
    velocity_pass = all(abs(r["velocity_error_mean_m_s"]) < 0.35 for r in episode_rows)
    episode_length_pass = all(r["sample_count"] == r["expected_sample_count"] for r in episode_rows)
    action_pass = all(
        r["left_action_min"] >= -1.05 and r["left_action_max"] <= 1.05 and r["right_action_min"] >= -1.05 and r["right_action_max"] <= 1.05
        for r in episode_rows
    )
    timestamp_pass = all(r["timestamp_strict"] and abs(r["mean_hz"] - 30.0) < 0.1 and r["max_dt_error_s"] < 1e-6 for r in episode_rows)
    finite_pass = all(r["finite"] for r in episode_rows)

    if not timestamp_pass:
        errors.append("timestamp validation failed")
    if not finite_pass:
        errors.append("non-finite data found")
    if not angle_pass:
        errors.append("angle branch flip found")
    if not gyro_pass:
        errors.append("angle-gyro consistency failed")
    if not gait_pass:
        errors.append("gait/contact periodicity failed")
    if not velocity_pass:
        errors.append("velocity tracking failed")
    if not episode_length_pass:
        errors.append("episode length validation failed")
    if not action_pass:
        errors.append("action validity failed")

    _write_csv(report_root / "episode_statistics.csv", episode_rows)
    _write_csv(report_root / "branch_flip_events.csv", branch_rows)
    _make_figures(raw_path, fig_dir)
    summary = {
        "raw_path": str(raw_path),
        "episode_count": len(episode_rows),
        "velocity_count": len(by_velocity),
        "ANGLE_CONTINUITY": "PASS" if angle_pass else "FAIL",
        "GYRO_CONSISTENCY": "PASS" if gyro_pass else "FAIL",
        "GAIT_PERIODICITY": "PASS" if gait_pass else "FAIL",
        "VELOCITY_TRACKING": "PASS" if velocity_pass else "FAIL",
        "EPISODE_LENGTH": "PASS" if episode_length_pass else "FAIL",
        "ACTION_VALIDITY": "PASS" if action_pass else "FAIL",
        "TIMESTAMPS": "PASS" if timestamp_pass else "FAIL",
        "FINITE_VALUES": "PASS" if finite_pass else "FAIL",
        "OVERALL_30HZ_VALIDATION": "PASS"
        if all([angle_pass, gyro_pass, gait_pass, velocity_pass, episode_length_pass, action_pass, timestamp_pass, finite_pass])
        else "FAIL",
        "errors": errors,
        "branch_flip_count": len(branch_rows),
        "episode_length_mismatch_count": int(
            sum(r["sample_count"] != r["expected_sample_count"] for r in episode_rows)
        ),
        "min_left_gyro_corr": float(min((r["left_gyro_corr"] for r in episode_rows), default=math.nan)),
        "min_right_gyro_corr": float(min((r["right_gyro_corr"] for r in episode_rows), default=math.nan)),
    }
    (report_root / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _write_report(report_root / "PHYSICAL_VALIDATION_REPORT.md", summary, episode_rows)
    return summary


def _make_figures(raw_path: Path, fig_dir: Path) -> None:
    with h5py.File(raw_path, "r") as h5:
        selected = [
            name
            for name, grp in h5["episodes"].items()
            if abs(float(grp.attrs["goal_velocity_m_s"]) - 1.30) < 1e-9
        ][:10]
        if not selected:
            selected = sorted(h5["episodes"].keys())[:10]
        plt.figure(figsize=(11, 6))
        for name in selected:
            grp = h5["episodes"][name]
            t = grp["time_s"][:] - grp["time_s"][0]
            plt.plot(t, np.degrees(grp["left_thigh_angle_rad"][:]), alpha=0.5, color="tab:blue")
            plt.plot(t, np.degrees(grp["right_thigh_angle_rad"][:]), alpha=0.5, color="tab:orange")
        plt.xlabel("time from episode start (s)")
        plt.ylabel("thigh angle (deg)")
        plt.title("1.30 m/s thigh angle overlay")
        plt.tight_layout()
        plt.savefig(fig_dir / "angle_overlay_1p30.png", dpi=150)
        plt.close()

        if selected:
            grp = h5["episodes"][selected[0]]
            t = grp["time_s"][:]
            dt = t - t[0]
            la = grp["left_thigh_angle_rad"][:]
            lg = grp["left_thigh_gyro_rad_s"][:]
            plt.figure(figsize=(11, 8))
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot(dt, np.degrees(la), label="left")
            ax1.plot(dt, np.degrees(grp["right_thigh_angle_rad"][:]), label="right")
            ax1.set_ylabel("angle deg")
            ax1.legend()
            ax2 = plt.subplot(4, 1, 2, sharex=ax1)
            ax2.plot(dt, np.degrees(lg), label="gyro")
            ax2.plot(dt, np.degrees(np.gradient(la, t)), label="dangle/dt")
            ax2.set_ylabel("deg/s")
            ax2.legend()
            ax3 = plt.subplot(4, 1, 3, sharex=ax1)
            ax3.plot(dt, grp["left_exo_action"][:], label="left action")
            ax3.plot(dt, grp["right_exo_action"][:], label="right action")
            ax3.legend()
            ax4 = plt.subplot(4, 1, 4, sharex=ax1)
            ax4.plot(dt, grp["actual_pelvis_velocity_m_s"][:], label="actual")
            ax4.plot(dt, grp["goal_velocity_m_s"][:], label="goal")
            ax4.step(dt, grp["left_foot_contact"][:].astype(float), where="post", label="L contact")
            ax4.step(dt, grp["right_foot_contact"][:].astype(float), where="post", label="R contact")
            ax4.legend()
            ax4.set_xlabel("time (s)")
            plt.tight_layout()
            plt.savefig(fig_dir / "episode_1p30_multichannel.png", dpi=150)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.plot(np.degrees(la), np.degrees(lg), ".-", markersize=3)
            plt.xlabel("left angle (deg)")
            plt.ylabel("left gyro (deg/s)")
            plt.title("Angle-gyro phase plot")
            plt.tight_layout()
            plt.savefig(fig_dir / "angle_gyro_phase_1p30.png", dpi=150)
            plt.close()


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 30 Hz Physical Validation Report",
        "",
        f"- ANGLE CONTINUITY: {summary['ANGLE_CONTINUITY']}",
        f"- GYRO CONSISTENCY: {summary['GYRO_CONSISTENCY']}",
        f"- GAIT PERIODICITY: {summary['GAIT_PERIODICITY']}",
        f"- VELOCITY TRACKING: {summary['VELOCITY_TRACKING']}",
        f"- EPISODE LENGTH: {summary['EPISODE_LENGTH']}",
        f"- ACTION VALIDITY: {summary['ACTION_VALIDITY']}",
        f"- OVERALL 30HZ VALIDATION: {summary['OVERALL_30HZ_VALIDATION']}",
        "",
        f"- episodes: {summary['episode_count']}",
        f"- branch flips: {summary['branch_flip_count']}",
        f"- episode length mismatches: {summary['episode_length_mismatch_count']}",
        f"- min left gyro corr: {summary['min_left_gyro_corr']}",
        f"- min right gyro corr: {summary['min_right_gyro_corr']}",
        "",
        "## Source Lines",
        "- angle unwrap implementation: `tcn/data_generation_v2/imu_kinematics.py:145-151`",
        "- gyro projection implementation: `tcn/data_generation_v2/imu_kinematics.py:128-146`",
        "- raw 30 Hz writer: `tcn/data_generation_v2/collect_teacher_30hz.py:242-334`",
    ]
    if summary["errors"]:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {e}" for e in summary["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
