"""Download NASA SMAP and MSL satellite telemetry datasets.

The SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory)
datasets contain labeled anomalies in satellite sensor telemetry data.

Source: https://github.com/khundman/telemanom
"""

import os
import urllib.request
import zipfile
import numpy as np


DATA_DIR = os.path.dirname(__file__)

# NASA telemanom dataset URLs
SMAP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data/SMAP.zip"
MSL_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data/MSL.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return

    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")


def download_dataset(dataset="SMAP"):
    """Download and extract SMAP or MSL dataset.

    Args:
        dataset: 'SMAP' or 'MSL'.
    """
    url = SMAP_URL if dataset == "SMAP" else MSL_URL
    zip_path = os.path.join(DATA_DIR, f"{dataset}.zip")
    extract_dir = os.path.join(DATA_DIR, dataset)

    if os.path.exists(extract_dir):
        print(f"{dataset} data already exists at {extract_dir}")
        return

    print(f"\nDownloading {dataset} dataset...")
    download_file(url, zip_path)

    print(f"Extracting {dataset}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    os.remove(zip_path)
    print(f"{dataset} extracted to {extract_dir}")


def download_labels():
    """Download anomaly labels CSV."""
    labels_path = os.path.join(DATA_DIR, "labeled_anomalies.csv")
    print("\nDownloading anomaly labels...")
    download_file(LABELS_URL, labels_path)
    return labels_path


def load_channel(dataset, channel, split="train"):
    """Load a single telemetry channel.

    Args:
        dataset: 'SMAP' or 'MSL'.
        channel: Channel name (e.g., 'T-1').
        split: 'train' or 'test'.

    Returns:
        np.ndarray of telemetry values.
    """
    path = os.path.join(DATA_DIR, dataset, f"{split}", f"{channel}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Channel data not found: {path}")
    return np.load(path)


def generate_synthetic_telemetry(n_channels=25, n_train=5000, n_test=2000, anomaly_ratio=0.05):
    """Generate synthetic satellite telemetry data for testing.

    Creates realistic-looking multivariate time series with injected anomalies
    simulating temperature, voltage, and gyroscope sensor data.

    Args:
        n_channels: Number of telemetry channels.
        n_train: Number of training time steps.
        n_test: Number of test time steps.
        anomaly_ratio: Fraction of test data that is anomalous.

    Returns:
        train_data, test_data, test_labels (np.ndarrays).
    """
    np.random.seed(42)

    # Generate base signals with different characteristics
    t_train = np.linspace(0, 50, n_train)
    t_test = np.linspace(50, 70, n_test)

    train_data = np.zeros((n_train, n_channels))
    test_data = np.zeros((n_test, n_channels))

    for ch in range(n_channels):
        freq = np.random.uniform(0.1, 2.0)
        amplitude = np.random.uniform(0.5, 3.0)
        offset = np.random.uniform(-1, 1)
        noise_level = np.random.uniform(0.01, 0.1)

        # Temperature-like channels (slow drift + seasonal)
        if ch < 8:
            train_data[:, ch] = (
                offset + amplitude * np.sin(2 * np.pi * freq * t_train / 10)
                + 0.1 * np.cumsum(np.random.randn(n_train)) / n_train
                + noise_level * np.random.randn(n_train)
            )
            test_data[:, ch] = (
                offset + amplitude * np.sin(2 * np.pi * freq * t_test / 10)
                + 0.1 * np.cumsum(np.random.randn(n_test)) / n_test
                + noise_level * np.random.randn(n_test)
            )
        # Voltage-like channels (step functions + noise)
        elif ch < 16:
            base_train = offset + np.random.choice([0, amplitude], n_train, p=[0.95, 0.05])
            base_test = offset + np.random.choice([0, amplitude], n_test, p=[0.95, 0.05])
            train_data[:, ch] = base_train + noise_level * np.random.randn(n_train)
            test_data[:, ch] = base_test + noise_level * np.random.randn(n_test)
        # Gyroscope-like channels (high-frequency oscillation)
        else:
            train_data[:, ch] = (
                amplitude * np.sin(2 * np.pi * freq * t_train)
                + 0.3 * amplitude * np.sin(2 * np.pi * 3 * freq * t_train)
                + noise_level * np.random.randn(n_train)
            )
            test_data[:, ch] = (
                amplitude * np.sin(2 * np.pi * freq * t_test)
                + 0.3 * amplitude * np.sin(2 * np.pi * 3 * freq * t_test)
                + noise_level * np.random.randn(n_test)
            )

    # Inject anomalies into test data
    test_labels = np.zeros(n_test, dtype=int)
    n_anomalies = int(n_test * anomaly_ratio)
    anomaly_starts = np.random.choice(n_test - 20, n_anomalies // 5, replace=False)

    for start in anomaly_starts:
        length = np.random.randint(5, 20)
        end = min(start + length, n_test)
        channels_affected = np.random.choice(n_channels, np.random.randint(1, 5), replace=False)

        anomaly_type = np.random.choice(["spike", "drift", "dropout"])
        for ch in channels_affected:
            if anomaly_type == "spike":
                test_data[start:end, ch] += np.random.uniform(3, 8) * np.std(train_data[:, ch])
            elif anomaly_type == "drift":
                drift = np.linspace(0, 5 * np.std(train_data[:, ch]), end - start)
                test_data[start:end, ch] += drift
            else:
                test_data[start:end, ch] = 0.0

        test_labels[start:end] = 1

    return train_data, test_data, test_labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NASA SMAP/MSL data")
    parser.add_argument("--dataset", choices=["SMAP", "MSL", "both", "synthetic"],
                        default="synthetic", help="Dataset to download")
    args = parser.parse_args()

    if args.dataset == "synthetic":
        print("Generating synthetic satellite telemetry data...")
        train, test, labels = generate_synthetic_telemetry()
        np.save(os.path.join(DATA_DIR, "train_data.npy"), train)
        np.save(os.path.join(DATA_DIR, "test_data.npy"), test)
        np.save(os.path.join(DATA_DIR, "test_labels.npy"), labels)
        print(f"Train shape: {train.shape}, Test shape: {test.shape}")
        print(f"Anomaly ratio: {labels.mean():.3f}")
        print(f"Data saved to {DATA_DIR}")
    else:
        if args.dataset in ("SMAP", "both"):
            download_dataset("SMAP")
        if args.dataset in ("MSL", "both"):
            download_dataset("MSL")
        download_labels()
