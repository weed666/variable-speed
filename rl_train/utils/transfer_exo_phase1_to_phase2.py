from __future__ import annotations

import argparse

from rl_train.utils.policy_transfer import transfer_exo_phase1_to_phase2_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer an Exo Phase 1 policy into an Exo Phase 2 initialization checkpoint."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    report = transfer_exo_phase1_to_phase2_checkpoint(
        source_checkpoint=args.source_checkpoint,
        target_config=args.target_config,
        output_checkpoint=args.output_checkpoint,
        report_path=args.report_path,
    )
    print("Exo Phase 1 -> Exo Phase 2 transfer complete.")
    print(f"Human max_abs_diff: {report['human_equivalence_max_abs_diff']:.3e}")
    print(f"Exo max_abs_diff: {report['exo_equivalence_max_abs_diff']:.3e}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
