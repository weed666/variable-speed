import os

import matplotlib.pyplot as plt
import numpy as np


def _as_float_series(raw_values):
    """Convert project JSON list values like [[x], ...] into a one-dimensional float array."""
    values = []
    for item in raw_values:
        if isinstance(item, list):
            if not item:
                values.append(np.nan)
            else:
                values.append(item[0])
        else:
            values.append(item)
    return np.asarray(values, dtype=float)


def _get_series_data(evaluation_data):
    """Return series_data from either a GaitData object or a loaded evaluation dict."""
    if hasattr(evaluation_data, "series_data"):
        return evaluation_data.series_data
    return evaluation_data.get("series_data", {})


def _get_metadata(evaluation_data):
    """Return metadata from either a GaitData object or a loaded evaluation dict."""
    if hasattr(evaluation_data, "metadata"):
        return evaluation_data.metadata
    return evaluation_data.get("metadata", {})


def extract_selected_reward_series(evaluation_data, reward_keys):
    """Extract selected per-step reward series from evaluation output."""
    series_data = _get_series_data(evaluation_data)
    metadata = _get_metadata(evaluation_data)
    reward_data = series_data.get("reward_data", {})
    missing_keys = metadata.get("reward_data", {}).get("missing_keys", {})
    target_data = series_data.get("target_data", {})
    sim_time_data = target_data.get("sim_time", [])

    extracted = {}
    warnings = []
    for reward_key in reward_keys:
        if reward_key not in reward_data:
            warnings.append(f"Reward key {reward_key!r} was not recorded; skipping plot.")
            continue
        if missing_keys.get(reward_key):
            warnings.append(
                f"Reward key {reward_key!r} was missing on some eval steps; skipping incomplete plot."
            )
            continue

        reward_values = _as_float_series(reward_data[reward_key])
        if len(sim_time_data) == len(reward_values):
            x_values = _as_float_series(sim_time_data)
            x_label = "sim_time"
        else:
            x_values = np.arange(len(reward_values), dtype=float)
            x_label = "eval step"
            if sim_time_data:
                warnings.append(
                    f"Reward key {reward_key!r} has no matching sim_time series; using eval step index."
                )

        extracted[reward_key] = {
            "x_values": x_values,
            "x_label": x_label,
            "reward_values": reward_values,
        }
    return extracted, warnings


def plot_reward_term_series(x_values, reward_values, reward_key, output_path, x_label):
    """Plot one per-step reward term without changing or accumulating values."""
    x_values = np.asarray(x_values, dtype=float)
    reward_values = np.asarray(reward_values, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(reward_values)
    if not np.any(finite_mask):
        print(f"Warning: reward key {reward_key!r} has no finite values; skipping plot.")
        return False

    finite_x = x_values[finite_mask]
    finite_rewards = reward_values[finite_mask]
    if not np.all(finite_mask):
        print(f"Warning: reward key {reward_key!r} contains NaN/Inf values; plotting finite samples only.")

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(finite_x, finite_rewards, linewidth=1.2)

    # Reward term plots preserve sparse event rewards; markers only highlight observed non-zero samples.
    event_mask = finite_rewards != 0.0
    if np.any(event_mask):
        ax.scatter(finite_x[event_mask], finite_rewards[event_mask], s=18, zorder=3)

    ax.set_title(reward_key)
    ax.set_xlabel(x_label)
    ax.set_ylabel("per-step reward value")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def generate_selected_reward_plots(evaluation_data, reward_keys, output_dir):
    """Generate standalone per-step plots for configured reward terms."""
    if not reward_keys:
        return []

    os.makedirs(output_dir, exist_ok=True)
    extracted, warnings = extract_selected_reward_series(evaluation_data, reward_keys)
    for warning in warnings:
        print(f"Warning: {warning}")

    generated_paths = []
    for reward_key, data in extracted.items():
        output_path = os.path.join(output_dir, f"{reward_key}_per_step.png")
        if plot_reward_term_series(
            data["x_values"],
            data["reward_values"],
            reward_key,
            output_path,
            data["x_label"],
        ):
            generated_paths.append(output_path)
    return generated_paths
