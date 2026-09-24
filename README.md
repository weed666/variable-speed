# Coming Soon

**Paper:** Coming soon.  
**arXiv:** Coming soon.  
**Project Page:** Coming soon.  
**Video:** Coming soon.

## Overview

Coming soon.

## Abstract

Coming soon.

## Installation

Clone the repository and install it in editable mode. MyoSuite is included as vendored source and must not be installed separately.

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
python -m pip install -e .
```

Optional dependency groups are available for analysis, hardware integration, and development:

```bash
python -m pip install -e '.[analysis]'
python -m pip install -e '.[hardware]'
python -m pip install -e '.[dev]'
```

## Repository Structure

```text
models/                 Musculoskeletal and exoskeleton simulation assets inherited from
                        MyoAssist, together with project-specific adaptations.
myosuite/               Vendored MyoSuite simulation code.
myoassist_utils/        Shared environment and simulation utilities.
rl_train/               Human/Exo reinforcement-learning pipeline.
tcn/                    TCN data generation, training, and deployment.
requirements.txt        Pinned base dependencies for the paper environment.
setup.py                Editable-install configuration and optional extras.
LICENSE                 License for original project contributions.
THIRD_PARTY_NOTICES.md  Notices for vendored and inherited material.
```

## Training and Usage

The general RL training entry point is:

```bash
python -m rl_train.run_train \
    --config_file_path <CONFIG>
```

The commands below form a sequential pipeline; transfer inputs are produced by the preceding training phase.

### Human Phase 1

```bash
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/human_phase_1.json
```

### Human Phase 1 → Human Phase 2 Transfer

```bash
python -m rl_train.utils.transfer_human_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Human_Phase_1/trained_models/model_29900800.zip \
    --target-config rl_train/train/train_configs/human_phase_2.json \
    --output-checkpoint rl_train/train/checkpoint/human_phase1_to_phase2/human_p2_init.zip \
    --report-path rl_train/train/checkpoint/human_phase1_to_phase2/transfer_report.json
```

The transferred checkpoint initializes Human Phase 2.

### Human Phase 2

```bash
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/human_phase_2.json
```

### Human Phase 2 → Exo Phase 1 Transfer

```bash
python -m rl_train.utils.transfer_human_phase2_to_exo_phase1 \
    --source-checkpoint rl_train/results/Human_Phase_2/trained_models/model_52445184.zip \
    --target-config rl_train/train/train_configs/human_phase2_to_exo_phase1_transfer.json \
    --output-checkpoint rl_train/train/checkpoint/human_phase2_to_exo_phase1/exo_p1_init.zip \
    --report-path rl_train/train/checkpoint/human_phase2_to_exo_phase1/transfer_report.json
```

The resulting initialization is used by the acceleration, deceleration, and steady Exo Phase 1 tasks.

### Exo Phase 1

```bash
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_1_ac.json

python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_1_de.json

python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_1_steady.json
```

### Exo Phase 1 → Exo Phase 2 Transfer

Acceleration:

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_ac/trained_models/model_31522816.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_ac.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_ac_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_ac.json
```

Deceleration:

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_de/trained_models/model_4784128.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_de.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_de_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_de.json
```

Steady:

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_steady/trained_models/model_4784128.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_steady.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_steady_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_steady.json
```

Each transferred checkpoint initializes the corresponding Exo Phase 2 task.

### Exo Phase 2

```bash
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_2_ac.json

python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_2_de.json

python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/exo_phase_2_steady.json
```

## TCN

### Data Generation

Generate, validate, resample, and package the steady-regime data:

```bash
python -m tcn.data_generation_v2.run_pipeline \
    --checkpoint rl_train/results/Exo_Phase_2_steady/trained_models/model_91815936.zip \
    --output-directory tcn/data_generation_v2/output
```

Generate and prepare the acceleration and deceleration regimes:

```bash
python -m tcn.data_generation_v2.acc_dec_pipeline \
    --acc-checkpoint rl_train/results/Exo_Phase_2_ac/trained_models/model_79757312.zip \
    --dec-checkpoint rl_train/results/Exo_Phase_2_de/trained_models/model_97648640.zip \
    --regime all \
    --output-dir tcn/data_generation_v2/output
```

Teacher actions are resampled from 30 Hz to 100 Hz using linear interpolation.

Prepare the unified 100 Hz dataset:

```bash
python -m tcn.data_generation_v2.prepare_unified_100hz_dataset \
    --accel-h5 <ACCELERATION_100HZ_H5> \
    --decel-h5 <DECELERATION_100HZ_H5> \
    --steady-h5 <STEADY_100HZ_H5> \
    --output-root tcn/data_generation_v2/output/unified_100hz_5500
```

### Training

```bash
python -m tcn.train \
    --config tcn/configs/train.json \
    --mode unified \
    --run-name <RUN_NAME>
```

### Deployment

The released deployment model is `tcn/deployment/unified_tcn_latest_100hz_deploy.pt`. A hardware-free dry run is available:

```bash
python -m tcn.deployment.pc_tcn_formal_controller \
    --model tcn/deployment/unified_tcn_latest_100hz_deploy.pt \
    --dry-run \
    --mock-imu \
    --duration 3 \
    --print-rate 2
```

To export a training checkpoint into the deployment format:

```bash
python -m tcn.deployment.export_deployment_checkpoint \
    --source-checkpoint <TRAINING_CHECKPOINT> \
    --output <DEPLOYMENT_CHECKPOINT>
```

## Acknowledgements

This project builds on MyoSuite and inherited MyoAssist simulation and model assets. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`models/PROVENANCE.md`](models/PROVENANCE.md) for provenance, licenses, and attribution.

## Citation

Coming soon.

## License

Original project code is released under the Apache License 2.0. See [`LICENSE`](LICENSE).

This repository also contains vendored and inherited third-party code and assets subject to their respective licenses and attribution requirements. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`models/PROVENANCE.md`](models/PROVENANCE.md) for details.
