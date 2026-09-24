# Unified TCN 100 Hz Deployment

Files:

- `pc_tcn_formal_controller.py`
- `unified_tcn_latest_100hz_deploy.pt`
- `export_deployment_checkpoint.py`

Runtime path:

dual thigh IMU -> standing calibration -> 100 Hz history -> checkpoint normalization -> causal TCN -> normalized command -> command-scale `12 Nm` -> clamp -> Teensy.

The model input is `[1, 4, 100]` with channel order:

1. left thigh angle rad
2. left thigh angular velocity rad/s
3. right thigh angle rad
4. right thigh angular velocity rad/s

Angles are standing-relative. Forward thigh flexion and forward angular motion must be positive after the configurable IMU signs:

- `--left-imu-direction`
- `--right-imu-direction`

Startup uses zero assistance until the 100-sample history is full. No zero padding is used.

The TCN output is a normalized exoskeleton command. The controller converts it to nominal command scale by:

`command_nm = action_norm * torque_scale_nm`

This is a command-scale actuator value, not a strict prediction of biological or measured hardware torque.

The raw TCN baseline keeps the slew limiter disabled. A CLI hook exists for future separately validated tests, but deployment baseline is:

`SLEW LIMITER: DISABLED`

Safe dry-run smoke:

```bash
python pc_tcn_formal_controller.py --dry-run --mock-imu --duration 3 --print-rate 2
```

Hardware mode still requires explicit `--arm`.

## Export utility

The exporter requires an explicit training checkpoint:

```bash
python -m tcn.deployment.export_deployment_checkpoint \
  --source-checkpoint <TRAINING_CHECKPOINT> \
  --output <DEPLOYMENT_CHECKPOINT>
```

The released `unified_tcn_latest_100hz_deploy.pt` is the authoritative hardware
artifact. The repository does not contain the exact training checkpoint needed
to reproduce that file byte-for-byte.
