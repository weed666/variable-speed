from __future__ import annotations

import json
from pathlib import Path

from .build_dataset import build_dataset
from .collect_teacher_30hz import collect
from .config import DEFAULT_OUTPUT, parse_args
from .inspect_dataset import final_audit
from .physical_validation import validate_raw_30hz
from .resample_100hz import resample_raw_to_100hz


def _final_audit_dir() -> Path:
    return DEFAULT_OUTPUT / "audits/final"


def _write_failed_final_audit(cfg, validation: dict) -> dict:
    audit_dir = _final_audit_dir()
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "OVERALL": "FAIL",
        "STOP_STAGE": "30hz_validation",
        "reason": "; ".join(validation.get("errors", ["30 Hz validation failed"])),
        "raw_path": str(cfg.raw_path),
        "processed_path": None,
        "dataset_dir": None,
        "ANGLE_CONTINUITY": validation.get("ANGLE_CONTINUITY", "FAIL"),
        "GYRO_CONSISTENCY": validation.get("GYRO_CONSISTENCY", "FAIL"),
        "GAIT_PERIODICITY": validation.get("GAIT_PERIODICITY", "FAIL"),
        "VELOCITY_TRACKING": validation.get("VELOCITY_TRACKING", "FAIL"),
        "EPISODE_LENGTH": validation.get("EPISODE_LENGTH", "FAIL"),
        "ACTION_VALIDITY": validation.get("ACTION_VALIDITY", "FAIL"),
        "IMU_RESAMPLING": "SKIPPED",
        "TEACHER_ACTION_LINEAR": "SKIPPED",
        "TRAIN_VAL_TEST_LEAKAGE": "SKIPPED",
        "SPEED_COVERAGE": "SKIPPED",
        "angle_branch_flips": validation.get("branch_flip_count", 0),
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (audit_dir / "TCN_DATASET_V2_AUDIT.md").write_text(
        "\n".join(
            [
                "# TCN Dataset V2 Audit",
                "",
                "- OVERALL: FAIL",
                "- STOP STAGE: 30hz_validation",
                f"- REASON: {summary['reason']}",
                f"- RAW DATA: {cfg.raw_path}",
                "- PROCESSED DATA: NOT GENERATED",
                "- DATASET: NOT GENERATED",
                f"- ANGLE CONTINUITY: {summary['ANGLE_CONTINUITY']}",
                f"- GYRO CONSISTENCY: {summary['GYRO_CONSISTENCY']}",
                f"- GAIT PERIODICITY: {summary['GAIT_PERIODICITY']}",
                f"- VELOCITY TRACKING: {summary['VELOCITY_TRACKING']}",
                f"- EPISODE LENGTH: {summary['EPISODE_LENGTH']}",
                f"- ACTION VALIDITY: {summary['ACTION_VALIDITY']}",
                "- IMU RESAMPLING: SKIPPED",
                "- TEACHER ACTION LINEAR INTERPOLATION: SKIPPED",
                "- TRAIN/VAL/TEST LEAKAGE: SKIPPED",
                f"- angle branch flips: {summary['angle_branch_flips']}",
                "",
                "100 Hz resampling and final TCN dataset construction were intentionally skipped because raw 30 Hz validation failed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("dataset_statistics.csv", "split_statistics.csv", "final_tcn_input_examples.csv"):
        (audit_dir / name).write_text("status,reason\nSKIPPED,30hz_validation_failed\n", encoding="utf-8")
    return summary


def run(argv: list[str] | None = None) -> dict:
    cfg = parse_args(argv)
    collection = collect(cfg)
    validation = validate_raw_30hz(cfg.raw_path)
    if validation["OVERALL_30HZ_VALIDATION"] != "PASS":
        audit = _write_failed_final_audit(cfg, validation)
        return {
            "TCN_DATA_GENERATION_V2": "FAIL",
            "stop_stage": "30hz_validation",
            "collection": collection,
            "validation": validation,
            "audit": audit,
        }
    processed = resample_raw_to_100hz(cfg.raw_path, cfg.processed_path, overwrite=cfg.overwrite)
    dataset = build_dataset(cfg.processed_path, cfg.dataset_dir, seed=cfg.seed, overwrite=cfg.overwrite)
    audit = final_audit(cfg.dataset_dir)
    return {
        "TCN_DATA_GENERATION_V2": audit["OVERALL"],
        "collection": collection,
        "validation": validation,
        "processed": processed,
        "dataset": dataset,
        "audit": audit,
    }


def _print_final(summary: dict, cfg) -> None:
    validation = summary.get("validation", {})
    audit = summary.get("audit", {})
    print(f"TCN DATA GENERATION V2: {summary.get('TCN_DATA_GENERATION_V2', 'FAIL')}")
    print("OLD PIPELINE REMOVED: YES")
    print(f"TEACHER CHECKPOINT: {cfg.checkpoint}")
    print("RAW FREQUENCY: 30 HZ")
    print("TARGET FREQUENCY: 100 HZ")
    print(f"VELOCITY RANGE: {cfg.min_velocity:.2f}-{cfg.max_velocity:.2f} M/S")
    print(f"VELOCITY STEP: {cfg.velocity_step:.2f} M/S")
    print(f"EPISODES PER VELOCITY: {cfg.effective_episodes_per_velocity()}")
    print(f"TOTAL VALID EPISODES: {validation.get('episode_count', 0) if validation.get('OVERALL_30HZ_VALIDATION') == 'PASS' else 0}")
    print(f"ANGLE BRANCH FLIPS: {validation.get('branch_flip_count', audit.get('angle_branch_flips', 0))}")
    print(f"ANGLE CONTINUITY: {validation.get('ANGLE_CONTINUITY', 'FAIL')}")
    print(f"GYRO CONSISTENCY: {validation.get('GYRO_CONSISTENCY', 'FAIL')}")
    print(f"GAIT PHYSIOLOGY: {validation.get('GAIT_PERIODICITY', 'FAIL')}")
    print(f"IMU RESAMPLING: {audit.get('IMU_RESAMPLING', 'FAIL')}")
    print(f"TEACHER ACTION LINEAR INTERPOLATION: {audit.get('TEACHER_ACTION_LINEAR', 'FAIL')}")
    print(f"TRAIN/VAL/TEST LEAKAGE: {audit.get('TRAIN_VAL_TEST_LEAKAGE', 'FAIL')}")
    print(f"RAW DATA: {cfg.raw_path}")
    processed_data = cfg.processed_path if summary.get("TCN_DATA_GENERATION_V2") == "PASS" else "NOT GENERATED"
    print(f"PROCESSED DATA: {processed_data}")
    print(f"FINAL REPORT: {_final_audit_dir() / 'TCN_DATASET_V2_AUDIT.md'}")


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    summary = run(argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _print_final(summary, cfg)
    return 0 if summary.get("TCN_DATA_GENERATION_V2") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
