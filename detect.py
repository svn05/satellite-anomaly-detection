"""Anomaly detection using trained LSTM-Autoencoder.

Computes reconstruction errors on test data and applies threshold-based
detection with smoothing to identify anomalous segments.

Usage:
    python detect.py
    python detect.py --threshold-sigma 3.5
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


def detect_anomalies(errors, threshold, min_length=3):
    """Detect anomalous segments in reconstruction errors.

    Args:
        errors: Reconstruction error array.
        threshold: Anomaly threshold.
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


def main():
    parser = argparse.ArgumentParser(description="Detect anomalies in satellite telemetry")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--threshold-sigma", type=float, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    data_dir = config["data"]["data_dir"]
    train_data = np.load(os.path.join(data_dir, "train_data.npy"))
    test_data = np.load(os.path.join(data_dir, "test_data.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

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

    # Compute errors
    print("Computing reconstruction errors on training data...")
    train_errors = compute_reconstruction_errors(model, train_windows, device)

    print("Computing reconstruction errors on test data...")
    test_errors = compute_reconstruction_errors(model, test_windows, device)

    # Smooth errors
    smooth_window = config["detection"]["smoothing_window"]
    test_errors_smooth = smooth_errors(test_errors, smooth_window)

    # Compute threshold
    sigma = args.threshold_sigma or config["detection"]["threshold_sigma"]
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

    # Align labels with windowed predictions
    window_labels = test_labels[window_size - 1 :]
    window_labels = window_labels[: len(predictions)]

    print(f"\nDetected {len(segments)} anomalous segments")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i + 1}: timesteps {start}-{end} (length={end - start})")

    print(f"\nResults saved to {results_dir}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Anomalous points: {predictions.sum()} ({predictions.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
