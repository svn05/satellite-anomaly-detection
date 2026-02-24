"""Training script for the LSTM-Autoencoder anomaly detector.

Trains on normal satellite telemetry data to learn reconstruction patterns.
Anomalies are then detected as high reconstruction error in test data.

Usage:
    python train.py
    python train.py --epochs 200 --batch-size 32
"""

import argparse
import os
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import LSTMAutoencoder, count_parameters


def load_config(config_path="configs/config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_windows(data, window_size, stride=1):
    """Create sliding windows from time series data.

    Args:
        data: np.ndarray of shape (n_timesteps, n_features).
        window_size: Length of each window.
        stride: Step size between windows.

    Returns:
        np.ndarray of shape (n_windows, window_size, n_features).
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
    return np.array(windows)


def normalize_data(train_data, test_data=None):
    """Z-score normalize using training statistics.

    Args:
        train_data: Training data array.
        test_data: Optional test data to normalize with same stats.

    Returns:
        Normalized arrays and (mean, std) stats.
    """
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std == 0] = 1.0  # Prevent division by zero

    train_normalized = (train_data - mean) / std
    if test_data is not None:
        test_normalized = (test_data - mean) / std
        return train_normalized, test_normalized, (mean, std)
    return train_normalized, (mean, std)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_x, in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        reconstruction, _ = model(batch_x)
        loss = criterion(reconstruction, batch_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    """Validate model on held-out data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, in dataloader:
            batch_x = batch_x.to(device)
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-Autoencoder")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data — try real NASA SMAP first, then fall back to synthetic
    data_dir = config["data"]["data_dir"]
    train_path = os.path.join(data_dir, "train_data.npy")
    test_path = os.path.join(data_dir, "test_data.npy")

    if not os.path.exists(train_path):
        print("Data not found. Downloading NASA SMAP multi-channel data...")
        try:
            from data.download_data import load_smap_multi_channel
            train_raw, test_raw, test_labels, ch_names, ch_types = load_smap_multi_channel(
                max_channels=config["model"]["input_dim"]
            )
            np.save(train_path, train_raw)
            np.save(test_path, test_raw)
            np.save(os.path.join(data_dir, "test_labels.npy"), test_labels)
            np.save(os.path.join(data_dir, "channel_names.npy"), np.array(ch_names))
            np.save(os.path.join(data_dir, "channel_types.npy"), np.array(ch_types))
        except Exception as e:
            print(f"Could not load SMAP data: {e}")
            print("Falling back to synthetic telemetry data...")
            from data.download_data import generate_synthetic_telemetry
            train_raw, test_raw, test_labels = generate_synthetic_telemetry()
            np.save(train_path, train_raw)
            np.save(test_path, test_raw)
            np.save(os.path.join(data_dir, "test_labels.npy"), test_labels)
    else:
        train_raw = np.load(train_path)
        test_raw = np.load(test_path)

    print(f"Train data shape: {train_raw.shape}")
    print(f"Test data shape: {test_raw.shape}")

    # Normalize
    train_norm, test_norm, stats = normalize_data(train_raw, test_raw)
    np.save(os.path.join(data_dir, "norm_stats.npy"), np.array([stats[0], stats[1]]))

    # Create windows
    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]
    train_windows = create_windows(train_norm, window_size, stride)
    print(f"Training windows: {train_windows.shape}")

    # Train/val split
    split = config["data"]["train_split"]
    n_train = int(len(train_windows) * split)
    train_set = train_windows[:n_train]
    val_set = train_windows[n_train:]

    train_tensor = torch.FloatTensor(train_set)
    val_tensor = torch.FloatTensor(val_set)

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = train_raw.shape[1]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
        seq_len=window_size,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Training setup
    epochs = args.epochs or config["training"]["epochs"]
    lr = args.lr or config["training"]["learning_rate"]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"]
    )

    if config["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["patience"]
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10} {'Time':>8}")
    print("-" * 54)

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6d} {train_loss:>12.6f} {val_loss:>12.6f} {current_lr:>10.6f} {elapsed:>7.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": config,
            }, os.path.join(model_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {os.path.join(model_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
