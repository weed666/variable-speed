# TCN Data Generation V2

Clean steady teacher data-generation pipeline.

## Pipeline

1. Collect raw 30 Hz teacher rollouts with continuous thigh orientation saved as raw rotation matrices/quaternions.
2. Compute thigh-mounted IMU sagittal angle by unwrapping per-episode raw femur body orientation relative to standing reference.
3. Validate raw 30 Hz timestamps, finite values, branch flips, angle-gyro consistency, gait/contact periodicity, velocity tracking, and action bounds.
4. Resample only after validation:
   - IMU angle/gyro and continuous state: linear interpolation to 100 Hz.
   - Teacher action and executed torque: linear interpolation to 100 Hz.
5. Build train/validation/test HDF5 splits by episode, balanced within each velocity.

## Example

```bash
python -m tcn.data_generation_v2.run_pipeline \
  --checkpoint rl_train/results/Exo_Phase_2_steady/trained_models/model_91815936.zip \
  --output-directory tcn/data_generation_v2/output \
  --minimum-velocity 0.90 \
  --maximum-velocity 1.60 \
  --velocity-step 0.05 \
  --episodes-per-velocity 100 \
  --episode-duration 5.0 \
  --random-seed 91815936
```

Use `--smoke-test --overwrite` for the required 2 episode per velocity smoke run.
