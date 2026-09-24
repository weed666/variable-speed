from __future__ import annotations

import argparse
import json

from rl_train.utils.policy_transfer import transfer_human_phase2_to_exo_phase1_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    report = transfer_human_phase2_to_exo_phase1_checkpoint(
        source_checkpoint=args.source_checkpoint,
        target_config=args.target_config,
        output_checkpoint=args.output_checkpoint,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
