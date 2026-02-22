"""Visualization tools for satellite telemetry anomaly detection.

Generates confidence-band plots, reconstruction error timelines,
and anomaly detection visualizations.

Usage:
    python visualize.py
"""

import argparse
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_reconstruction_with_confidence(
    original, reconstructed, errors, threshold, labels=None,
    channel_idx=0, title="Telemetry Reconstruction with Confidence Bands",
    save_path=None
):
    """Plot original vs reconstructed signal with confidence bands.

    Args:
        original: Original test data (n_timesteps, n_features).
        reconstructed: Reconstructed data (n_timesteps, n_features).
        errors: Per-timestep reconstruction errors.
        threshold: Anomaly threshold value.
        labels: Ground truth anomaly labels (optional).
        channel_idx: Which channel to plot.
        title: Plot title.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    timesteps = np.arange(len(original))

    # Plot 1: Original vs Reconstructed with confidence band
    ax1 = axes[0]
    ax1.plot(timesteps, original[:, channel_idx], "b-", alpha=0.7, label="Original", linewidth=0.8)
    ax1.plot(timesteps, reconstructed[:, channel_idx], "r-", alpha=0.5, label="Reconstructed", linewidth=0.8)

    # Confidence band (mean ± 2*std of reconstruction error per channel)
    residuals = np.abs(original[:, channel_idx] - reconstructed[:, channel_idx])
    mean_res = np.convolve(residuals, np.ones(20) / 20, mode="same")
    std_res = np.convolve(residuals ** 2, np.ones(20) / 20, mode="same")
    std_res = np.sqrt(np.maximum(std_res - mean_res ** 2, 0))

    upper = reconstructed[:, channel_idx] + 2 * std_res
    lower = reconstructed[:, channel_idx] - 2 * std_res
    ax1.fill_between(timesteps, lower, upper, alpha=0.15, color="red", label="95% Confidence Band")

    ax1.set_ylabel("Signal Value")
    ax1.set_title(f"{title} — Channel {channel_idx}")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reconstruction error with threshold
    ax2 = axes[1]
    ax2.plot(timesteps[:len(errors)], errors, "k-", alpha=0.7, linewidth=0.6, label="Reconstruction Error")
    ax2.axhline(y=threshold, color="r", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.4f})")

    # Highlight anomalous regions
    anomaly_mask = errors > threshold
    ax2.fill_between(
        timesteps[:len(errors)], 0, errors,
        where=anomaly_mask, color="red", alpha=0.3, label="Detected Anomaly"
    )

    ax2.set_ylabel("MSE Error")
    ax2.set_title("Reconstruction Error")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Ground truth vs predictions
    ax3 = axes[2]
    if labels is not None:
        aligned_labels = labels[:len(errors)]
        predictions = anomaly_mask.astype(int)

        ax3.fill_between(timesteps[:len(aligned_labels)], 0, aligned_labels,
                         alpha=0.3, color="blue", label="Ground Truth")
        ax3.fill_between(timesteps[:len(predictions)], 0, predictions * 0.8,
                         alpha=0.3, color="red", label="Predictions")

        ax3.set_ylim(-0.1, 1.5)
        ax3.set_ylabel("Anomaly")
        ax3.set_title("Ground Truth vs Predictions")
        ax3.legend(loc="upper right", fontsize=8)
    else:
        ax3.fill_between(timesteps[:len(errors)], 0, anomaly_mask.astype(float),
                         alpha=0.3, color="red", label="Detected")
        ax3.set_ylabel("Anomaly")
        ax3.set_title("Detected Anomalies")
        ax3.legend(loc="upper right", fontsize=8)

    ax3.set_xlabel("Timestep")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_error_distribution(train_errors, test_errors, threshold, save_path=None):
    """Plot distribution of reconstruction errors for train and test data.

    Args:
        train_errors: Training set reconstruction errors.
        test_errors: Test set reconstruction errors.
        threshold: Anomaly threshold.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(train_errors, bins=100, alpha=0.6, color="blue", label="Train (Normal)", density=True)
    ax.hist(test_errors, bins=100, alpha=0.6, color="orange", label="Test", density=True)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_multi_channel(data, errors, threshold, labels=None, n_channels=6, save_path=None):
    """Plot multiple telemetry channels with anomaly highlighting.

    Args:
        data: Telemetry data (n_timesteps, n_features).
        errors: Per-window reconstruction errors.
        threshold: Anomaly threshold.
        labels: Ground truth labels.
        n_channels: Number of channels to display.
        save_path: Path to save figure.
    """
    n_channels = min(n_channels, data.shape[1])
    fig, axes = plt.subplots(n_channels + 1, 1, figsize=(16, 3 * (n_channels + 1)), sharex=True)

    timesteps = np.arange(len(data))
    anomaly_mask = np.zeros(len(data), dtype=bool)
    anomaly_mask[:len(errors)] = errors > threshold

    channel_types = ["Temperature"] * 8 + ["Voltage"] * 8 + ["Gyroscope"] * 9
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

    for i in range(n_channels):
        ax = axes[i]
        ch_type = channel_types[i] if i < len(channel_types) else "Sensor"
        ax.plot(timesteps, data[:, i], color=colors[i % len(colors)], linewidth=0.6, alpha=0.8)
        ax.fill_between(timesteps, data[:, i].min(), data[:, i].max(),
                        where=anomaly_mask[:len(data)], color="red", alpha=0.15)
        ax.set_ylabel(f"Ch {i}\n({ch_type})", fontsize=8)
        ax.grid(True, alpha=0.2)

    # Bottom plot: errors
    ax = axes[-1]
    ax.plot(np.arange(len(errors)), errors, "k-", linewidth=0.5, alpha=0.7)
    ax.axhline(y=threshold, color="r", linestyle="--", linewidth=1.5)
    ax.fill_between(np.arange(len(errors)), 0, errors,
                    where=errors > threshold, color="red", alpha=0.3)
    ax.set_ylabel("Error")
    ax.set_xlabel("Timestep")
    ax.grid(True, alpha=0.2)

    plt.suptitle("Multi-Channel Satellite Telemetry with Anomaly Detection", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize anomaly detection results")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["data_dir"]
    results_dir = config["output"]["results_dir"]
    plots_dir = config["output"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    test_data = np.load(os.path.join(data_dir, "test_data.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    test_errors = np.load(os.path.join(results_dir, "test_errors.npy"))
    test_errors_smooth = np.load(os.path.join(results_dir, "test_errors_smooth.npy"))

    # Compute threshold
    train_data = np.load(os.path.join(data_dir, "train_data.npy"))
    from train import create_windows, normalize_data
    from detect import compute_reconstruction_errors, compute_threshold
    import torch

    train_norm, test_norm, _ = normalize_data(train_data, test_data)
    train_errors_raw = np.load(os.path.join(results_dir, "test_errors.npy"))

    sigma = config["detection"]["threshold_sigma"]
    threshold = np.mean(test_errors) + sigma * np.std(test_errors)

    # Generate plots
    print("Generating visualization plots...\n")

    plot_error_distribution(
        test_errors[:len(test_errors) // 2],
        test_errors,
        threshold,
        save_path=os.path.join(plots_dir, "error_distribution.png"),
    )

    plot_multi_channel(
        test_data, test_errors_smooth, threshold, test_labels,
        n_channels=6,
        save_path=os.path.join(plots_dir, "multi_channel.png"),
    )

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
