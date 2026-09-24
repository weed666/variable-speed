#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REGIMES = ("acceleration", "deceleration", "steady")
HISTORY_STEPS = 100
INPUT_MAP = {
    "left_thigh_angle_rad": "left_thigh_angle_rad",
    "left_thigh_angular_velocity_rad_s": "left_thigh_angular_velocity_rad_s",
    "right_thigh_angle_rad": "right_thigh_angle_rad",
    "right_thigh_angular_velocity_rad_s": "right_thigh_angular_velocity_rad_s",
}
TARGET_MAP = {
    "left_teacher_action_norm": "left_teacher_action_norm",
    "right_teacher_action_norm": "right_teacher_action_norm",
}
TORQUE_MAP = {
    "left_teacher_torque_nm": "left_teacher_torque_nm",
    "right_teacher_torque_nm": "right_teacher_torque_nm",
}
FIELD_MAP = {
    **INPUT_MAP,
    **TARGET_MAP,
    **TORQUE_MAP,
    "timestamp_s": "timestamp_s",
    "task_phase": "task_phase",
}


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def as_py(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(group: h5py.Group) -> str:
    h = hashlib.sha256()
    keys = [
        "timestamp_s",
        *INPUT_MAP.values(),
        *TARGET_MAP.values(),
        *TORQUE_MAP.values(),
    ]
    for key in keys:
        if key in group:
            h.update(key.encode("utf-8"))
            h.update(np.ascontiguousarray(group[key][()]).view(np.uint8))
    return h.hexdigest()


def finite_range(values: list[float]) -> list[float | None]:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return [None, None]
    return [min(vals), max(vals)]


def read_rows(regime: str, h5_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        names = sorted(h5["episodes"].keys())
        root_attrs = {k: as_py(v) for k, v in h5.attrs.items()}
        for name in names:
            grp = h5["episodes"][name]
            attrs = {k: as_py(v) for k, v in grp.attrs.items()}
            status = str(attrs.get("collection_status", "valid"))
            term = str(attrs.get("termination_json", ""))
            fall = '"fall": true' in term.lower() or str(attrs.get("fall", "")).lower() == "true"
            required = ["timestamp_s", *INPUT_MAP.values(), *TARGET_MAP.values(), *TORQUE_MAP.values()]
            missing = [key for key in required if key not in grp]
            length = int(grp["timestamp_s"].shape[0]) if "timestamp_s" in grp else 0
            digest = content_digest(grp)
            task_id = str(attrs.get("task_index", attrs.get("task_id", regime)))
            initial_velocity = float(attrs.get("initial_velocity_m_s", attrs.get("initial_velocity", attrs.get("goal_velocity_m_s", 0.0))))
            final_velocity = float(attrs.get("goal_velocity_m_s", 0.0))
            acceleration = float(attrs.get("signed_acceleration_m_s2", attrs.get("target_acceleration", 0.0)))
            ramp_duration = float(attrs.get("ramp_duration_s", attrs.get("ramp_end_time_s", 0.0))) - float(attrs.get("ramp_start_time_s", 0.0))
            if ramp_duration < 0:
                ramp_duration = 0.0
            row = {
                "regime": regime,
                "source_h5_file": str(h5_path),
                "source_group": f"/episodes/{name}",
                "h5_group": f"/episodes/{name}",
                "episode_group": name,
                "episode_id": str(attrs.get("episode_id", name)),
                "trajectory_hash": str(attrs.get("trajectory_hash", "")),
                "content_digest": digest,
                "task_id": task_id,
                "initial_velocity": initial_velocity,
                "goal_velocity": final_velocity,
                "target_acceleration": acceleration,
                "acceleration_magnitude": abs(acceleration),
                "ramp_duration": ramp_duration,
                "num_samples": length,
                "valid_windows": max(0, length - HISTORY_STEPS + 1),
                "status": status,
                "fall": fall,
                "missing_required_keys": missing,
            }
            if status != "valid" or fall or missing or row["valid_windows"] <= 0:
                failures.append(row)
            else:
                rows.append(row)
    meta = {
        "root_attrs": root_attrs,
        "total_episode_groups": len(rows) + len(failures),
        "valid_episode_count": len(rows),
        "excluded_episode_count": len(failures),
        "excluded_examples": failures[:20],
    }
    return rows, meta


def bind_components(regime: str, rows: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    dsu = DSU(len(rows))
    bindings: dict[str, dict[str, list[int]]] = {"episode_id": defaultdict(list), "trajectory_task": defaultdict(list), "content_digest": defaultdict(list)}
    for i, row in enumerate(rows):
        if row["episode_id"]:
            bindings["episode_id"][row["episode_id"]].append(i)
        task_key = (
            row["trajectory_hash"],
            round(float(row["initial_velocity"]), 6),
            round(float(row["goal_velocity"]), 6),
            round(float(row["acceleration_magnitude"]), 6),
            round(float(row["ramp_duration"]), 6),
            row["task_id"],
        )
        if row["trajectory_hash"]:
            bindings["trajectory_task"][repr(task_key)].append(i)
        if row["content_digest"]:
            bindings["content_digest"][row["content_digest"]].append(i)
    for groups in bindings.values():
        for indices in groups.values():
            if len(indices) > 1:
                first = indices[0]
                for idx in indices[1:]:
                    dsu.union(first, idx)
    by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_root[dsu.find(i)].append(row)
    duplicate_ids = {
        key: [rows[i]["episode_group"] for i in idxs]
        for key, idxs in bindings["episode_id"].items()
        if len(idxs) > 1
    }
    duplicate_digest_groups = sum(1 for idxs in bindings["content_digest"].values() if len(idxs) > 1)
    audit = {
        "regime": regime,
        "component_count": len(by_root),
        "max_component_size": max((len(v) for v in by_root.values()), default=0),
        "duplicate_episode_ids": duplicate_ids,
        "duplicate_content_digest_group_count": duplicate_digest_groups,
    }
    return list(by_root.values()), audit


def steady_stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    return (round(float(row["goal_velocity"]), 6),)


def dynamic_stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(row["initial_velocity"]), 6),
        round(float(row["goal_velocity"]), 6),
        round(float(row["acceleration_magnitude"]), 6),
        round(float(row["ramp_duration"]), 6),
        row["task_id"],
    )


def split_components(
    regime: str,
    components: list[list[dict[str, Any]]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    strata: dict[tuple[Any, ...], list[list[dict[str, Any]]]] = defaultdict(list)
    for comp in components:
        key = steady_stratum(comp[0]) if regime == "steady" else dynamic_stratum(comp[0])
        strata[key].append(comp)
    out = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        comps = list(strata[key])
        rng.shuffle(comps)
        n = sum(len(c) for c in comps)
        targets = {
            "train": int(round(n * train_ratio)),
            "val": int(round(n * val_ratio)),
        }
        counts = {"train": 0, "val": 0}
        for comp in comps:
            if counts["train"] + len(comp) <= targets["train"]:
                out["train"].extend(comp)
                counts["train"] += len(comp)
            elif counts["val"] + len(comp) <= targets["val"]:
                out["val"].extend(comp)
                counts["val"] += len(comp)
            else:
                out["test"].extend(comp)
    for split in out:
        out[split] = sorted(out[split], key=lambda r: r["episode_group"])
    return out


def leakage_check(regime_splits: dict[str, dict[str, list[dict[str, Any]]]], all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"per_regime": {}, "overall_pass": True}
    for regime, splits in regime_splits.items():
        all_groups = {r["episode_group"] for r in all_rows[regime]}
        split_groups = {s: {r["episode_group"] for r in rows} for s, rows in splits.items()}
        split_ids = {s: {r["episode_id"] for r in rows if r["episode_id"]} for s, rows in splits.items()}
        split_traj = {
            s: {
                (
                    r["trajectory_hash"],
                    round(float(r["initial_velocity"]), 6),
                    round(float(r["goal_velocity"]), 6),
                    round(float(r["acceleration_magnitude"]), 6),
                    round(float(r["ramp_duration"]), 6),
                    r["task_id"],
                )
                for r in rows
                if r["trajectory_hash"]
            }
            for s, rows in splits.items()
        }
        split_digest = {s: {r["content_digest"] for r in rows if r["content_digest"]} for s, rows in splits.items()}
        assigned = set().union(*split_groups.values())
        duplicate_assigned_count = sum(len(v) for v in split_groups.values()) - len(assigned)
        checks = {
            "group_train_val_intersection": len(split_groups["train"] & split_groups["val"]),
            "group_train_test_intersection": len(split_groups["train"] & split_groups["test"]),
            "group_val_test_intersection": len(split_groups["val"] & split_groups["test"]),
            "episode_id_train_val_intersection": sorted(split_ids["train"] & split_ids["val"])[:20],
            "episode_id_train_test_intersection": sorted(split_ids["train"] & split_ids["test"])[:20],
            "episode_id_val_test_intersection": sorted(split_ids["val"] & split_ids["test"])[:20],
            "trajectory_task_train_val_intersection_count": len(split_traj["train"] & split_traj["val"]),
            "trajectory_task_train_test_intersection_count": len(split_traj["train"] & split_traj["test"]),
            "trajectory_task_val_test_intersection_count": len(split_traj["val"] & split_traj["test"]),
            "content_digest_train_val_intersection_count": len(split_digest["train"] & split_digest["val"]),
            "content_digest_train_test_intersection_count": len(split_digest["train"] & split_digest["test"]),
            "content_digest_val_test_intersection_count": len(split_digest["val"] & split_digest["test"]),
            "all_valid_episodes_assigned": assigned == all_groups,
            "missing_episode_groups": sorted(all_groups - assigned)[:20],
            "extra_episode_groups": sorted(assigned - all_groups)[:20],
            "duplicate_assignment_count": duplicate_assigned_count,
        }
        regime_pass = (
            checks["group_train_val_intersection"] == 0
            and checks["group_train_test_intersection"] == 0
            and checks["group_val_test_intersection"] == 0
            and not checks["episode_id_train_val_intersection"]
            and not checks["episode_id_train_test_intersection"]
            and not checks["episode_id_val_test_intersection"]
            and checks["trajectory_task_train_val_intersection_count"] == 0
            and checks["trajectory_task_train_test_intersection_count"] == 0
            and checks["trajectory_task_val_test_intersection_count"] == 0
            and checks["content_digest_train_val_intersection_count"] == 0
            and checks["content_digest_train_test_intersection_count"] == 0
            and checks["content_digest_val_test_intersection_count"] == 0
            and checks["all_valid_episodes_assigned"]
            and checks["duplicate_assignment_count"] == 0
        )
        result["per_regime"][regime] = {**checks, "pass": regime_pass}
        result["overall_pass"] = bool(result["overall_pass"] and regime_pass)
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_mapped_h5(source_path: Path, mapped_path: Path, regime: str) -> None:
    if mapped_path.exists() or mapped_path.is_symlink():
        mapped_path.unlink()
    mapped_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as src, h5py.File(mapped_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["schema_version"] = "unified_100hz_external_link_schema_v1"
        dst.attrs["source_processed_100hz"] = str(source_path)
        dst.attrs["regime_name"] = regime
        dst.attrs["field_mapping_json"] = json.dumps(FIELD_MAP, sort_keys=True)
        root = dst.create_group("episodes")
        for name, src_grp in src["episodes"].items():
            out = root.create_group(name)
            for key, value in src_grp.attrs.items():
                out.attrs[key] = value
            for dst_key, src_key in FIELD_MAP.items():
                if src_key not in src_grp:
                    raise KeyError(f"{source_path} /episodes/{name} missing required source field {src_key}")
                out[dst_key] = h5py.ExternalLink(str(source_path), f"/episodes/{name}/{src_key}")


def compute_normalization(splits: dict[str, dict[str, list[dict[str, Any]]]], raw_links: dict[str, Path]) -> dict[str, Any]:
    per_regime = {}
    balanced_means = []
    balanced_stds = []
    raw_values = []
    for regime in REGIMES:
        count = 0
        sums = np.zeros(len(INPUT_MAP), dtype=np.float64)
        sumsq = np.zeros(len(INPUT_MAP), dtype=np.float64)
        with h5py.File(raw_links[regime], "r") as h5:
            for row in splits[regime]["train"]:
                g = h5[row["h5_group"].strip("/")]
                arr = np.stack([np.asarray(g[src][:], dtype=np.float64) for src in INPUT_MAP.keys()], axis=1)
                count += arr.shape[0]
                sums += arr.sum(axis=0)
                sumsq += np.square(arr).sum(axis=0)
        mean = sums / max(count, 1)
        std = np.sqrt(np.maximum(sumsq / max(count, 1) - mean * mean, 1e-12))
        per_regime[regime] = {"input_mean": mean.tolist(), "input_std": std.tolist(), "frame_count": int(count)}
        balanced_means.append(mean)
        balanced_stds.append(std)
        raw_values.append((count, mean, std))
    balanced_mean = np.mean(np.stack(balanced_means, axis=0), axis=0)
    balanced_std = np.mean(np.stack(balanced_stds, axis=0), axis=0)
    total_count = sum(c for c, _m, _s in raw_values)
    raw_mean = sum(c * m for c, m, _s in raw_values) / max(total_count, 1)
    second = sum(c * (s * s + m * m) for c, m, s in raw_values) / max(total_count, 1)
    raw_std = np.sqrt(np.maximum(second - raw_mean * raw_mean, 1e-12))
    return {
        "schema_version": "unified_100hz_symlink_dataset_normalization_v1",
        "train_only": True,
        "input_channel_order": list(INPUT_MAP.keys()),
        "input_source_channel_order": list(INPUT_MAP.values()),
        "per_regime_train": per_regime,
        "unified_train_regime_balanced": {"input_mean": balanced_mean.tolist(), "input_std": balanced_std.tolist()},
        "unified_train_raw_frame_weighted": {"input_mean": raw_mean.tolist(), "input_std": raw_std.tolist()},
        "units": {"angle": "rad", "angular_velocity": "rad/s"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare symlink-backed Unified 100 Hz TCN dataset splits.")
    parser.add_argument("--accel-h5", required=True, type=Path)
    parser.add_argument("--decel-h5", required=True, type=Path)
    parser.add_argument("--steady-h5", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} exists and is non-empty; pass --overwrite to replace it")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    raw_dir = output / "raw"
    split_dir = output / "splits"
    meta_dir = output / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "acceleration": args.accel_h5.resolve(),
        "deceleration": args.decel_h5.resolve(),
        "steady": args.steady_h5.resolve(),
    }
    raw_links = {
        "acceleration": raw_dir / "acceleration.h5",
        "deceleration": raw_dir / "deceleration.h5",
        "steady": raw_dir / "steady.h5",
    }

    all_rows: dict[str, list[dict[str, Any]]] = {}
    source_meta: dict[str, Any] = {}
    components_audit: dict[str, Any] = {}
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for regime in REGIMES:
        rows, meta = read_rows(regime, source_paths[regime])
        all_rows[regime] = rows
        source_meta[regime] = meta
        components, comp_audit = bind_components(regime, rows)
        components_audit[regime] = comp_audit
        splits[regime] = split_components(regime, components, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        write_mapped_h5(source_paths[regime], raw_links[regime], regime)

    leakage = leakage_check(splits, all_rows)
    if not leakage["overall_pass"]:
        write_json(output / "split_audit.json", {"leakage_check": leakage, "component_audit": components_audit})
        raise RuntimeError(f"Split leakage check failed; see {output / 'split_audit.json'}")

    split_manifest: dict[str, Any] = {
        "schema_version": "unified_100hz_symlink_episode_split_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "split_policy": {
            "steady": "group by goal_velocity_m_s plus trajectory_hash-derived components; 70/15/15 before windowing",
            "acceleration_deceleration": "stratified by initial velocity, final velocity, acceleration magnitude, ramp duration, task id; union-find binds duplicate episode_id, trajectory/task hash, and content digest",
        },
        "splits": {},
    }
    summary_rows = []
    for regime in REGIMES:
        split_json = {"schema_version": "unified_100hz_symlink_loader_split_v1", "regime": regime, "episodes": {}}
        for split_name in ("train", "val", "test"):
            records = []
            for idx, row in enumerate(splits[regime][split_name]):
                records.append({
                    "h5_file": str(raw_links[regime]),
                    "h5_group": row["h5_group"],
                    "episode_id": row["episode_id"],
                    "global_episode_uid": f"{regime}_{split_name}_{idx:06d}",
                    "task_id": row["task_id"],
                    "target_acceleration": row["target_acceleration"],
                    "initial_velocity": row["initial_velocity"],
                    "goal_velocity": row["goal_velocity"],
                    "trajectory_hash": row["trajectory_hash"],
                    "content_digest": row["content_digest"],
                    "source_h5_file": row["source_h5_file"],
                    "source_group": row["source_group"],
                    "num_samples": row["num_samples"],
                    "valid_windows": row["valid_windows"],
                })
            split_json["episodes"][split_name] = records
            split_manifest["splits"].setdefault(regime, {})[split_name] = {
                "episode_count": len(records),
                "window_count": sum(r["valid_windows"] for r in records),
            }
            summary_rows.append({
                "regime": regime,
                "split": split_name,
                "episode_count": len(records),
                "window_count": sum(r["valid_windows"] for r in records),
            })
        write_json(split_dir / f"{regime}_split.json", split_json)

    normalization = compute_normalization(splits, raw_links)
    write_json(meta_dir / "normalization_stats.json", normalization)
    source_files = {
            regime: {
                "source_h5": str(source_paths[regime]),
                "mapped_h5": str(raw_links[regime]),
                "sha256": sha256_file(source_paths[regime]),
                **source_meta[regime],
        }
        for regime in REGIMES
    }
    manifest = {
        "schema_version": "unified_100hz_external_link_dataset_manifest_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_root": str(output),
        "history_steps": HISTORY_STEPS,
        "sample_rate_hz": 100,
        "source_files": source_files,
        "input_channel_order": list(INPUT_MAP.keys()),
        "input_source_channel_order": list(INPUT_MAP.values()),
        "target_channel_order": list(TARGET_MAP.keys()),
        "target_source_channel_order": list(TARGET_MAP.values()),
        "field_mapping": FIELD_MAP,
        "label_semantics": "teacher normalized exoskeleton action linearly interpolated from saved 30 Hz raw Teacher samples; Nm torque channel is linearly interpolated executed torque",
        "torque_scale_nm": 12.0,
        "component_audit": components_audit,
        "split_manifest": split_manifest,
        "leakage_check": leakage,
    }
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "split_manifest.json", split_manifest)
    write_json(output / "split_audit.json", {"leakage_check": leakage, "component_audit": components_audit})
    write_csv(output / "split_summary.csv", summary_rows)
    print(json.dumps({
        "dataset_root": str(output),
        "episode_counts": {r: source_meta[r]["valid_episode_count"] for r in REGIMES},
        "split_counts": {
            r: {s: split_manifest["splits"][r][s]["episode_count"] for s in ("train", "val", "test")}
            for r in REGIMES
        },
        "leakage_pass": leakage["overall_pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
