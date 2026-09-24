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

from .collect_teacher_30hz import EpisodeSpec, load_teacher_config, run_episode
from .complete_raw_30hz import RAW_KEYS, _copy_episode, _copy_root_attrs, _write_payload_group, _payload_arrays, is_complete_episode, sha256_file
from .config import DEFAULT_OUTPUT, REPO_ROOT, TEACHER_HZ
from .imu_kinematics import build_imu_definition


REPORT_ROOT = DEFAULT_OUTPUT / "reports/dedup_refill"
VELOCITIES = [v / 100.0 for v in range(90, 161, 5)]


def trajectory_hash_arrays(
    goal_velocity: float,
    left_angle: np.ndarray,
    left_gyro: np.ndarray,
    right_angle: np.ndarray,
    right_gyro: np.ndarray,
    left_action: np.ndarray,
    right_action: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(np.asarray([round(float(goal_velocity), 2)], dtype=np.float64).tobytes())
    for arr in (left_angle, left_gyro, right_angle, right_gyro, left_action, right_action):
        h.update(np.asarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def trajectory_hash_group(grp: h5py.Group) -> str:
    v = float(grp.attrs["goal_velocity_m_s"])
    return trajectory_hash_arrays(
        v,
        grp["left_thigh_angle_rad"][:],
        grp["left_thigh_gyro_rad_s"][:],
        grp["right_thigh_angle_rad"][:],
        grp["right_thigh_gyro_rad_s"][:],
        grp["left_exo_action"][:],
        grp["right_exo_action"][:],
    )


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


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def audit_duplicates(raw_path: Path) -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    clusters: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    keep_groups: list[str] = []
    existing_hashes_by_speed: dict[float, set[str]] = {v: set() for v in VELOCITIES}
    with h5py.File(raw_path, "r") as h5:
        by_speed_hash: dict[tuple[float, str], list[str]] = {}
        for name, grp in h5["episodes"].items():
            if not is_complete_episode(grp):
                continue
            v = round(float(grp.attrs["goal_velocity_m_s"]), 2)
            h = trajectory_hash_group(grp)
            by_speed_hash.setdefault((v, h), []).append(name)

        for v in VELOCITIES:
            keys = [key for key in by_speed_hash if key[0] == round(v, 2)]
            total = sum(len(by_speed_hash[key]) for key in keys)
            unique = len(keys)
            duplicate_clusters = sum(1 for key in keys if len(by_speed_hash[key]) > 1)
            duplicate_episodes = sum(len(by_speed_hash[key]) - 1 for key in keys if len(by_speed_hash[key]) > 1)
            counts.append(
                {
                    "goal_velocity_m_s": f"{v:.2f}",
                    "total_complete_episodes": total,
                    "unique_hashes": unique,
                    "duplicate_hash_clusters": duplicate_clusters,
                    "duplicate_episodes": duplicate_episodes,
                }
            )

        for (v, h), names in sorted(by_speed_hash.items()):
            names_sorted = sorted(names)
            keep = names_sorted[0]
            keep_groups.append(keep)
            existing_hashes_by_speed.setdefault(v, set()).add(h)
            if len(names_sorted) <= 1:
                continue
            cluster_id = f"v{int(round(v * 100)):03d}_{h[:12]}"
            episode_ids = []
            reference_indices = []
            random_seeds = []
            for name in names_sorted:
                grp = h5["episodes"][name]
                episode_ids.append(_decode_attr(grp.attrs.get("episode_id", name)))
                reference_indices.append(str(grp.attrs.get("reference_index", "")))
                random_seeds.append(str(grp.attrs.get("random_seed", "")))
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "goal_velocity_m_s": f"{v:.2f}",
                    "trajectory_hash": h,
                    "episode_groups": "|".join(names_sorted),
                    "episode_ids": "|".join(episode_ids),
                    "reference_indices": "|".join(reference_indices),
                    "random_seeds": "|".join(random_seeds),
                    "kept_group": keep,
                    "removed_count": len(names_sorted) - 1,
                }
            )
            for name in names_sorted[1:]:
                grp = h5["episodes"][name]
                removed.append(
                    {
                        "cluster_id": cluster_id,
                        "removed_group": name,
                        "kept_group": keep,
                        "goal_velocity_m_s": f"{v:.2f}",
                        "trajectory_hash": h,
                        "episode_id": _decode_attr(grp.attrs.get("episode_id", name)),
                        "reference_index": grp.attrs.get("reference_index", ""),
                        "random_seed": grp.attrs.get("random_seed", ""),
                        "remove_reason": "exact_duplicate_trajectory_hash; kept lowest episode group id",
                    }
                )

    _write_csv(REPORT_ROOT / "duplicate_clusters.csv", clusters)
    _write_csv(REPORT_ROOT / "removed_duplicate_episodes.csv", removed)
    _write_csv(REPORT_ROOT / "unique_episode_counts_by_speed.csv", counts)
    original_complete = sum(int(r["total_complete_episodes"]) for r in counts)
    original_unique = sum(int(r["unique_hashes"]) for r in counts)
    report = [
        "# Dedup Audit Before",
        "",
        f"- Active raw: `{raw_path}`",
        f"- Original complete episodes: {original_complete}",
        f"- Original unique trajectory hashes: {original_unique}",
        f"- Duplicate episodes to remove: {len(removed)}",
        "",
        "Hash fields: goal velocity, left/right thigh angle, left/right thigh gyro, left/right normalized exo action.",
        "Only complete 150-sample `time_limit` episodes are considered for the active unique set.",
    ]
    (REPORT_ROOT / "DEDUP_AUDIT_BEFORE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "keep_groups": keep_groups,
        "existing_hashes_by_speed": existing_hashes_by_speed,
        "counts": counts,
        "clusters": clusters,
        "removed": removed,
        "original_complete": original_complete,
        "original_unique": original_unique,
        "duplicate_removed": len(removed),
    }


def _candidate_hash(payload: dict[str, Any], imu: Any) -> tuple[str, dict[str, np.ndarray]]:
    arrays = _payload_arrays(payload, imu)
    v = float(payload["spec"]["velocity_m_s"])
    h = trajectory_hash_arrays(
        v,
        arrays["left_angle"],
        arrays["left_gyro"],
        arrays["right_angle"],
        arrays["right_gyro"],
        arrays["left_action"],
        arrays["right_action"],
    )
    return h, arrays


def validate_candidate(payload: dict[str, Any], imu: Any, existing_hashes: set[str]) -> tuple[bool, str, str | None]:
    rows = payload["rows"]
    if payload["termination"].get("reason") != "time_limit":
        return False, "termination_not_time_limit", None
    if len(rows) != 150:
        return False, f"sample_count_{len(rows)}", None
    h, arrays = _candidate_hash(payload, imu)
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
    left_corr = float(np.corrcoef(np.gradient(arrays["left_angle"], 1.0 / TEACHER_HZ), arrays["left_gyro"])[0, 1])
    right_corr = float(np.corrcoef(np.gradient(arrays["right_angle"], 1.0 / TEACHER_HZ), arrays["right_gyro"])[0, 1])
    if left_corr <= 0.95 or right_corr <= 0.95:
        return False, f"angle_gyro_corr_left_{left_corr:.3f}_right_{right_corr:.3f}", None
    if np.max(np.abs(arrays["left_action"])) > 1.0 + 1e-6 or np.max(np.abs(arrays["right_action"])) > 1.0 + 1e-6:
        return False, "action_out_of_range", None
    if np.max(np.abs(arrays["left_torque"] - 12.0 * arrays["left_action"])) > 1e-6:
        return False, "left_torque_scale", None
    if np.max(np.abs(arrays["right_torque"] - 12.0 * arrays["right_action"])) > 1e-6:
        return False, "right_torque_scale", None
    if h in existing_hashes:
        return False, "duplicate_trajectory_hash", h
    return True, "accepted", h


def _perturbation_for_attempt(seed: int, attempt: int, used_refs_available: bool) -> dict[str, Any] | None:
    if used_refs_available and attempt < 90:
        return None
    scale_level = max(1, attempt // 90)
    return {
        "type": "small_initial_qpos_qvel_gaussian",
        "seed": int(seed + 17),
        "qpos_scale": float(min(0.002, 0.00005 * scale_level)),
        "qvel_scale": float(min(0.02, 0.0005 * scale_level)),
        "excluded_qpos_joints": ["pelvis_tx", "pelvis_ty"],
        "excluded_qvel_joints": ["pelvis_tx"],
    }


def dedup_refill(raw_path: Path, checkpoint: Path, *, seed_base: int = 291815936, max_attempts_per_speed: int = 5000) -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = audit_duplicates(raw_path)
    backup_path = raw_path.with_name("steady_30hz.before_dedup_refill.h5")
    tmp_path = raw_path.with_name("steady_30hz.dedup_refill.tmp.h5")
    if not backup_path.exists():
        shutil.copy2(raw_path, backup_path)
    active_sha = sha256_file(raw_path)
    backup_sha = sha256_file(backup_path)
    if backup_sha != active_sha:
        raise RuntimeError("dedup backup SHA does not match active raw; refusing to continue")

    base_config, _base_config_dict = load_teacher_config(checkpoint)
    imu = build_imu_definition(base_config.env_params.model_path)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    refill_stats: list[dict[str, Any]] = []

    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(raw_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        _copy_root_attrs(src, dst)
        dst.attrs["dedup_refill_backup_sha256"] = backup_sha
        dst.attrs["dedup_refill_source_raw"] = str(raw_path)
        dst.attrs["trajectory_hash_fields"] = "goal_velocity,left_thigh_angle,left_thigh_gyro,right_thigh_angle,right_thigh_gyro,left_exo_action,right_exo_action"
        root = dst.create_group("episodes")
        out_idx = 0
        for name in sorted(audit["keep_groups"]):
            _copy_episode(src["episodes"][name], root, f"{out_idx:06d}")
            root[f"{out_idx:06d}"].attrs["source_group"] = name
            out_idx += 1

        used_refs_by_speed: dict[float, set[int]] = {v: set() for v in VELOCITIES}
        for name in audit["keep_groups"]:
            grp = src["episodes"][name]
            v = round(float(grp.attrs["goal_velocity_m_s"]), 2)
            used_refs_by_speed.setdefault(v, set()).add(int(grp.attrs.get("reference_index", -1)))

        existing_hashes_by_speed: dict[float, set[str]] = {float(k): set(v) for k, v in audit["existing_hashes_by_speed"].items()}
        for v in VELOCITIES:
            v = round(v, 2)
            existing = existing_hashes_by_speed.setdefault(v, set())
            needed = 100 - len(existing)
            accepted = 0
            attempts = 0
            while accepted < needed:
                if attempts >= max_attempts_per_speed:
                    raise RuntimeError(f"failed to collect {needed} unique replacements for {v:.2f} m/s after {attempts} attempts")
                unused_refs = [idx for idx in range(90) if idx not in used_refs_by_speed.setdefault(v, set())]
                if attempts < len(unused_refs):
                    ref = unused_refs[attempts]
                    perturbation = None
                else:
                    ref = attempts % 90
                    perturbation = _perturbation_for_attempt(seed_base + int(round(v * 100)) * 100_000 + attempts, attempts, False)
                seed = int(seed_base + int(round(v * 100)) * 100_000 + attempts)
                spec = EpisodeSpec(
                    episode_id=f"steady_v{int(round(v * 100)):03d}_dedup_refill_{accepted:04d}",
                    velocity_m_s=float(v),
                    seed=seed,
                    reference_index=int(ref),
                    duration_s=5.0,
                    initial_state_perturbation=perturbation,
                )
                print(
                    f"[dedup-refill] v={v:.2f} accepted={accepted}/{needed} attempt={attempts} seed={seed} ref={ref} perturb={'yes' if perturbation else 'no'}",
                    flush=True,
                )
                payload = run_episode(base_config, checkpoint, spec, imu)
                ok, reason, traj_hash = validate_candidate(payload, imu, existing)
                attempts += 1
                if not ok:
                    rejected_rows.append(
                        {
                            "goal_velocity_m_s": f"{v:.2f}",
                            "seed": seed,
                            "reference_index": ref,
                            "perturbation_json": json.dumps(perturbation, sort_keys=True),
                            "reason": reason,
                            "sample_count": len(payload["rows"]),
                            "termination_reason": payload["termination"].get("reason"),
                            "trajectory_hash": traj_hash or "",
                        }
                    )
                    continue
                _write_payload_group(root, f"{out_idx:06d}", payload, imu)
                root[f"{out_idx:06d}"].attrs["initial_state_perturbation_json"] = json.dumps(perturbation, sort_keys=True)
                root[f"{out_idx:06d}"].attrs["trajectory_hash"] = str(traj_hash)
                existing.add(str(traj_hash))
                used_refs_by_speed[v].add(int(ref))
                accepted_rows.append(
                    {
                        "episode_group": f"{out_idx:06d}",
                        "goal_velocity_m_s": f"{v:.2f}",
                        "seed": seed,
                        "reference_index": ref,
                        "perturbation_json": json.dumps(perturbation, sort_keys=True),
                        "trajectory_hash": traj_hash,
                    }
                )
                accepted += 1
                out_idx += 1
            refill_stats.append(
                {
                    "goal_velocity_m_s": f"{v:.2f}",
                    "unique_before_refill": 100 - needed,
                    "needed": needed,
                    "accepted": accepted,
                    "attempts": attempts,
                    "unique_after_refill": len(existing),
                }
            )

    _write_csv(REPORT_ROOT / "refilled_episode_statistics.csv", refill_stats)
    _write_csv(REPORT_ROOT / "accepted_unique_refill_episodes.csv", accepted_rows)
    _write_csv(REPORT_ROOT / "rejected_refill_candidates.csv", rejected_rows)
    summary = {
        "raw_path": str(raw_path),
        "tmp_path": str(tmp_path),
        "backup_path": str(backup_path),
        "backup_sha256": backup_sha,
        "original_complete": audit["original_complete"],
        "original_unique": audit["original_unique"],
        "duplicate_removed": audit["duplicate_removed"],
        "new_unique_collected": len(accepted_rows),
        "rejected_candidates": len(rejected_rows),
    }
    (REPORT_ROOT / "dedup_refill_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-base", type=int, default=291815936)
    p.add_argument("--max-attempts-per-speed", type=int, default=5000)
    args = p.parse_args(argv)
    summary = dedup_refill(
        args.raw_path.resolve(),
        args.checkpoint.resolve(),
        seed_base=int(args.seed_base),
        max_attempts_per_speed=int(args.max_attempts_per_speed),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
