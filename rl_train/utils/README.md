# Checkpoint Transfer

These utilities create initialization checkpoints between the Human and Exo
training phases. Run every command from the repository root. Use a new output
path: the utilities reject overwriting the source checkpoint.

## Transfer overview

```text
Human Phase 1
    ↓
Human Phase 2
    ↓
Exo Phase 1
    ↓
Exo Phase 2
    ├── Steady
    ├── Acceleration
    └── Deceleration
```

| Transfer | Module | Input policy | Output policy | Main structural change |
|---|---|---|---|---|
| Human Phase 1 → Human Phase 2 | `transfer_human_phase1_to_phase2` | `HumanActorCriticPolicy`, 44 observations, 22 actions | `HumanActorCriticPolicy`, 46 observations, 22 actions | Adds `target_acceleration` and `goal_velocity`. |
| Human Phase 2 → Exo Phase 1 | `transfer_human_phase2_to_exo_phase1` | `HumanActorCriticPolicy`, 46 observations, 22 actions | `HumanExoActorCriticPolicy`, 52 observations, 24 actions | Preserves the Human actor and adds a two-action Exo actor. |
| Exo Phase 1 → Exo Phase 2 | `transfer_exo_phase1_to_phase2` | `HumanExoActorCriticPolicy`, 52 observations, 24 actions | `HumanExoActorCriticPolicy`, 54 observations, 24 actions | Adds applied Exo commands to the Human/critic observations and unfreezes the Human side. |

All three modules use the same CLI schema:

```text
--source-checkpoint
--target-config
--output-checkpoint
--report-path
```

The commands below are the repository-relative commands used to generate the
release initialization checkpoints.

## 1. Human Phase 1 → Human Phase 2

Human Phase 2 appends two scalar observations to the 44-dimensional Human
Phase 1 observation, in this order:

1. `target_acceleration`
2. `goal_velocity`

`target_velocity` already exists in Human Phase 1. The actor first layer changes
from `[64, 44]` to `[64, 46]`: the original 44 columns are copied and the two
new columns are zero-initialized. The remaining Human actor parameters and all
22 `log_std` values are copied. The action space remains 22-dimensional.

The target 46-input critic is freshly initialized. Source optimizer,
rollout-buffer, and scheduler state are not copied. The transfer does not freeze
the Human actor, critic, or `log_std`.

```bash
python -m rl_train.utils.transfer_human_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Human_Phase_1/trained_models/model_29900800.zip \
    --target-config rl_train/train/train_configs/human_phase_2.json \
    --output-checkpoint rl_train/train/checkpoint/human_phase1_to_phase2/human_p2_init.zip \
    --report-path rl_train/train/checkpoint/human_phase1_to_phase2/transfer_report.json
```

## 2. Human Phase 2 → Exo Phase 1

The source is a 46-observation, 22-action Human policy. The target is a
Human/Exo policy with 52 total observations and 24 actions:

- Human actor: observations `0:46`, actions `0:22`.
- Exo actor: observations `0:52`, actions `22:24` (`Exo_R`, `Exo_L`).
- Common critic: observations `0:52`.
- Observations `46:52`: three-step bilateral normalized Exo-action history.

All Human actor layers are copied. Human `log_std[0:22]` is copied. The Exo
actor, Exo `log_std[22:24]`, and common critic keep their fresh target-policy
initialization. Source optimizer, rollout-buffer, and scheduler state are not
copied. The target config freezes the Human actor and Human `log_std` slice
while Exo Phase 1 trains.

`human_phase2_to_exo_phase1_transfer.json` is used only as the shared target-policy
construction config for this conversion. It is not the formal Acceleration,
Deceleration, or Steady Exo Phase 1 training config.

```bash
python -m rl_train.utils.transfer_human_phase2_to_exo_phase1 \
    --source-checkpoint rl_train/results/Human_Phase_2/trained_models/model_52445184.zip \
    --target-config rl_train/train/train_configs/human_phase2_to_exo_phase1_transfer.json \
    --output-checkpoint rl_train/train/checkpoint/human_phase2_to_exo_phase1/exo_p1_init.zip \
    --report-path rl_train/train/checkpoint/human_phase2_to_exo_phase1/transfer_report.json
```

## 3. Exo Phase 1 → Exo Phase 2

The same transfer implementation is applied independently to Steady,
Acceleration, and Deceleration.

Exo Phase 2 appends the two currently applied normalized Exo commands,
producing 54 total observations. The Human actor
input changes from 46 to 48: its original columns are copied and the two new
columns are zero-initialized. The Exo actor still consumes observations `0:52`
and is copied directly. The action space remains 22 Human plus 2 Exo actions.

The 54-input critic is freshly initialized. All 24 `log_std` values are copied.
The target configs leave the Human actor and `log_std` trainable. Source
optimizer, rollout-buffer, and scheduler state are not copied. If the source
checkpoint has an adjacent `session_config.json`, its Exo output-filter settings
are checked against the target config; historical metadata is read but never
rewritten.

### Steady

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_steady/trained_models/model_4784128.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_steady.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_steady_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_steady.json
```

### Acceleration

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_ac/trained_models/model_31522816.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_ac.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_ac_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_ac.json
```

### Deceleration

```bash
python -m rl_train.utils.transfer_exo_phase1_to_phase2 \
    --source-checkpoint rl_train/results/Exo_Phase_1_de/trained_models/model_4784128.zip \
    --target-config rl_train/train/train_configs/exo_phase_2_de.json \
    --output-checkpoint rl_train/train/checkpoint/exo_phase1_to_phase2/exo_p2_de_init.zip \
    --report-path rl_train/train/checkpoint/exo_phase1_to_phase2/transfer_report_de.json
```

Each successful conversion writes a new SB3 PPO checkpoint and a JSON report
containing the copied keys, dimensions, and equivalence checks. All five
commands above completed successfully for this release.
