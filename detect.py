"""Anomaly detection using trained LSTM-Autoencoder.

Computes reconstruction errors on test data and applies threshold-based
detection with per-channel anomaly scoring, adaptive confidence bands,
and dynamic threshold tuning.

Usage:
    python detect.py
    python detect.py --threshold-sigma 3.5
    python detect.py --adaptive
"""

import argparse
import os
import numpy as np
import torch
import yaml

from model import LSTMAutoencoder
from train import create_windows, normalize_data


def load_model(model_path, config, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = LSTMAutoencoder(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        seq_len=config["data"]["window_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_reconstruction_errors(model, data_windows, device, batch_size=64):
    """Compute per-window reconstruction errors.

    Args:
        model: Trained LSTM-Autoencoder.
        data_windows: np.ndarray of shape (n_windows, seq_len, n_features).
        device: torch device.
        batch_size: Inference batch size.

    Returns:
        np.ndarray of reconstruction errors (n_windows,).
    """
    model.eval()
    errors = []

    tensor_data = torch.FloatTensor(data_windows)
    for i in range(0, len(tensor_data), batch_size):
        batch = tensor_data[i : i + batch_size].to(device)
        with torch.no_grad():
            recon, _ = model(batch)
            batch_errors = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)


def compute_per_channel_errors(model, data_windows, device, batch_size=64):
    """Compute per-channel reconstruction errors for root cause analysis.

    Instead of aggregating across all channels, returns error per channel
    to identify which sensors are contributing most to detected anomalies.

    Args:
        model: Trained LSTM-Autoencoder.
        data_windows: (n_windows, seq_len, n_features).
        device: torch device.
        batch_size: Inference batch size.

    Returns:
        channel_errors: (n_windows, n_features) — per-channel MSE.
    """
    model.eval()
    all_channel_errors = []

    tensor_data = torch.FloatTensor(data_windows)
    for i in range(0, len(tensor_data), batch_size):
        batch = tensor_data[i : i + batch_size].to(device)
        with torch.no_grad():
            recon, _ = model(batch)
            # MSE per channel: average over time, keep channels separate
            ch_errors = torch.mean((batch - recon) ** 2, dim=1)  # (batch, n_features)
            all_channel_errors.append(ch_errors.cpu().numpy())

    return np.concatenate(all_channel_errors, axis=0)


def smooth_errors(errors, window_size=5):
    """Apply moving average smoothing to reconstruction errors."""
    if window_size <= 1:
        return errors
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(errors, kernel, mode="same")
    return smoothed


def compute_threshold(train_errors, sigma=3.0):
    """Compute anomaly threshold as mean + sigma * std of training errors.

    Args:
        train_errors: Reconstruction errors on training data.
        sigma: Number of standard deviations above mean.

    Returns:
        threshold: Anomaly threshold value.
    """
    mean = np.mean(train_errors)
    std = np.std(train_errors)
    threshold = mean + sigma * std
    return threshold


def compute_adaptive_threshold(errors, window_size=200, sigma=3.0, min_samples=50):
    """Compute dynamic threshold that adapts to local error statistics.

    Uses a rolling window to compute local mean and std, producing a
    threshold that adjusts to non-stationary behavior in the telemetry.

    Args:
        errors: Reconstruction error array (n_windows,).
        window_size: Rolling window size for local statistics.
        sigma: Number of std deviations above local mean.
        min_samples: Minimum samples before using adaptive threshold.

    Returns:
        thresholds: (n_windows,) — per-timestep adaptive threshold.
    """
    n = len(errors)
    thresholds = np.zeros(n)

    # Global fallback for early timesteps
    global_mean = np.mean(errors)
    global_std = np.std(errors)
    global_threshold = global_mean + sigma * global_std

    for i in range(n):
        if i < min_samples:
            thresholds[i] = global_threshold
        else:
            start = max(0, i - window_size)
            local = errors[start:i]
            local_mean = np.mean(local)
            local_std = np.std(local)
            # Blend local and global to prevent overfitting to recent anomalies
            blend = min(1.0, len(local) / window_size)
            blended_mean = blend * local_mean + (1 - blend) * global_mean
            blended_std = blend * local_std + (1 - blend) * global_std
            thresholds[i] = blended_mean + sigma * blended_std

    return thresholds


def detect_anomalies(errors, threshold, min_length=3):
    """Detect anomalous segments in reconstruction errors.

    Args:
        errors: Reconstruction error array.
        threshold: Anomaly threshold (scalar or array for adaptive).
        min_length: Minimum consecutive anomalous points.

    Returns:
        predictions: Binary array (1=anomaly, 0=normal).
        anomaly_segments: List of (start, end) tuples.
    """
    raw_predictions = (errors > threshold).astype(int)

    # Filter short anomalies
    predictions = np.zeros_like(raw_predictions)
    i = 0
    anomaly_segments = []
    while i < len(raw_predictions):
        if raw_predictions[i] == 1:
            j = i
            while j < len(raw_predictions) and raw_predictions[j] == 1:
                j += 1
            if j - i >= min_length:
                predictions[i:j] = 1
                anomaly_segments.append((i, j))
            i = j
        else:
            i += 1

    return predictions, anomaly_segments


def rank_anomalous_channels(channel_errors, predictions, channel_names=None):
    """Rank channels by their contribution to detected anomalies.

    For each detected anomaly segment, identifies which channels have
    the highest reconstruction error — useful for root cause analysis.

    Args:
        channel_errors: (n_windows, n_channels) per-channel errors.
        predictions: (n_windows,) binary predictions.
        channel_names: Optional list of channel names.

    Returns:
        List of dicts with channel rankings per anomaly segment.
    """
    n_channels = channel_errors.shape[1]
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    # Global channel baseline (mean error on normal points)
    normal_mask = predictions == 0
    if normal_mask.sum() > 0:
        normal_means = channel_errors[normal_mask].mean(axis=0)
    else:
        normal_means = channel_errors.mean(axis=0)

    # Find anomaly segments
    segments = []
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            j = i
            while j < len(predictions) and predictions[j] == 1:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1

    rankings = []
    for start, end in segments:
        seg_errors = channel_errors[start:end].mean(axis=0)
        # Anomaly score = how much above normal baseline
        anomaly_scores = seg_errors / (normal_means + 1e-8)

        ranked_indices = np.argsort(anomaly_scores)[::-1]
        ranking = {
            "segment": (start, end),
            "length": end - start,
            "channels": [
                {
                    "name": channel_names[idx],
                    "score": float(anomaly_scores[idx]),
                    "error": float(seg_errors[idx]),
                    "baseline": float(normal_means[idx]),
                }
                for idx in ranked_indices[:5]  # Top 5 contributors
            ],
        }
        rankings.append(ranking)

    return rankings


def main():
    parser = argparse.ArgumentParser(description="Detect anomalies in satellite telemetry")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--threshold-sigma", type=float, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive (dynamic) thresholding")
    parser.add_argument("--adaptive-window", type=int, default=200,
                        help="Rolling window size for adaptive threshold")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    data_dir = config["data"]["data_dir"]
    train_data = np.load(os.path.join(data_dir, "train_data.npy"))
    test_data = np.load(os.path.join(data_dir, "test_data.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Load channel names if available
    channel_names_path = os.path.join(data_dir, "channel_names.npy")
    if os.path.exists(channel_names_path):
        channel_names = list(np.load(channel_names_path, allow_pickle=True))
    else:
        channel_names = [f"ch_{i}" for i in range(train_data.shape[1])]

    # Update config with actual input dim
    config["model"]["input_dim"] = train_data.shape[1]

    # Normalize
    train_norm, test_norm, _ = normalize_data(train_data, test_data)

    # Create windows
    window_size = config["data"]["window_size"]
    train_windows = create_windows(train_norm, window_size)
    test_windows = create_windows(test_norm, window_size)

    # Load model
    model_path = args.model_path or os.path.join(config["output"]["model_dir"], "best_model.pt")
    model = load_model(model_path, config, device)
    print(f"Loaded model from {model_path}")

    # Compute aggregate errors
    print("Computing reconstruction errors on training data...")
    train_errors = compute_reconstruction_errors(model, train_windows, device)

    print("Computing reconstruction errors on test data...")
    test_errors = compute_reconstruction_errors(model, test_windows, device)

    # Compute per-channel errors
    print("Computing per-channel reconstruction errors...")
    channel_errors = compute_per_channel_errors(model, test_windows, device)

    # Smooth errors
    smooth_window = config["detection"]["smoothing_window"]
    test_errors_smooth = smooth_errors(test_errors, smooth_window)

    # Compute threshold (static or adaptive)
    sigma = args.threshold_sigma or config["detection"]["threshold_sigma"]

    if args.adaptive:
        print(f"\nUsing adaptive threshold (σ={sigma}, window={args.adaptive_window})")
        threshold = compute_adaptive_threshold(
            test_errors_smooth, window_size=args.adaptive_window, sigma=sigma
        )
    else:
        threshold = compute_threshold(train_errors, sigma)
        print(f"\nAnomaly threshold (σ={sigma}): {threshold:.6f}")

    # Detect anomalies
    min_length = config["detection"]["min_anomaly_length"]
    predictions, segments = detect_anomalies(test_errors_smooth, threshold, min_length)

    # Save results
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "test_errors.npy"), test_errors)
    np.save(os.path.join(results_dir, "test_errors_smooth.npy"), test_errors_smooth)
    np.save(os.path.join(results_dir, "predictions.npy"), predictions)
    np.save(os.path.join(results_dir, "channel_errors.npy"), channel_errors)
    np.save(os.path.join(results_dir, "train_errors.npy"), train_errors)
    if isinstance(threshold, np.ndarray):
        np.save(os.path.join(results_dir, "adaptive_thresholds.npy"), threshold)

    # Print detection summary
    print(f"\nDetected {len(segments)} anomalous segments")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i + 1}: timesteps {start}-{end} (length={end - start})")

    # Per-channel anomaly ranking
    print(f"\n--- Per-Channel Anomaly Scoring ---")
    rankings = rank_anomalous_channels(channel_errors, predictions, channel_names)
    for r in rankings[:5]:  # Show top 5 segments
        seg = r["segment"]
        print(f"\n  Segment {seg[0]}-{seg[1]} (length={r['length']}):")
        for ch in r["channels"][:3]:
            print(f"    {ch['name']:>6s}: score={ch['score']:.2f}x baseline (error={ch['error']:.6f})")

    print(f"\nResults saved to {results_dir}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Anomalous points: {predictions.sum()} ({predictions.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
