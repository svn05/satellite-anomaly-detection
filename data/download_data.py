"""Download NASA SMAP and MSL satellite telemetry datasets.

The SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory)
datasets contain labeled anomalies in satellite sensor telemetry data.

Source: https://github.com/khundman/telemanom
"""

import csv
import json
import os
import urllib.request
import zipfile
import numpy as np


DATA_DIR = os.path.dirname(__file__)

# NASA telemanom dataset URLs (primary: full archive, fallback: individual zips)
DATA_ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
SMAP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data/SMAP.zip"
MSL_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data/MSL.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

# Curated channel selection by sensor type for multi-channel analysis
SMAP_CHANNELS = {
    "temperature": ["T-1", "T-2", "T-3", "T-4", "T-5", "T-8", "T-9", "T-12"],
    "power": ["P-1"],
    "ground_systems": ["G-1", "G-2", "G-3", "G-4", "G-6", "G-7"],
    "data_handling": ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6", "D-7", "D-8"],
    "antenna": ["A-1", "A-2", "A-3", "A-4"],
    "energy": ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8"],
}


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

    Tries the full data.zip archive first (more reliable), then
    falls back to individual dataset zips.

    Args:
        dataset: 'SMAP' or 'MSL'.
    """
    extract_dir = os.path.join(DATA_DIR, dataset)

    if os.path.exists(extract_dir):
        print(f"{dataset} data already exists at {extract_dir}")
        return

    # Try full data archive first (more reliable URL)
    full_zip_path = os.path.join(DATA_DIR, "data.zip")
    try:
        print(f"\nDownloading NASA telemanom data archive...")
        download_file(DATA_ZIP_URL, full_zip_path)
        print(f"Extracting...")
        with zipfile.ZipFile(full_zip_path, "r") as z:
            z.extractall(DATA_DIR)
        if os.path.exists(full_zip_path):
            os.remove(full_zip_path)
        if os.path.exists(extract_dir):
            print(f"{dataset} extracted to {extract_dir}")
            return
    except Exception as e:
        print(f"  Full archive failed: {e}")
        if os.path.exists(full_zip_path):
            os.remove(full_zip_path)

    # Fallback: individual dataset zip
    url = SMAP_URL if dataset == "SMAP" else MSL_URL
    zip_path = os.path.join(DATA_DIR, f"{dataset}.zip")

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


def parse_anomaly_labels(labels_path, spacecraft="SMAP"):
    """Parse labeled_anomalies.csv into a dict of channel -> anomaly ranges.

    Args:
        labels_path: Path to labeled_anomalies.csv.
        spacecraft: 'SMAP' or 'MSL'.

    Returns:
        dict mapping channel_id -> list of (start, end) anomaly ranges.
    """
    anomaly_info = {}
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["spacecraft"] == spacecraft:
                ranges = json.loads(row["anomaly_sequences"])
                anomaly_info[row["chan_id"]] = ranges
    return anomaly_info


def load_channel(dataset, channel, split="train"):
    """Load a single telemetry channel.

    Args:
        dataset: 'SMAP' or 'MSL'.
        channel: Channel name (e.g., 'T-1').
        split: 'train' or 'test'.

    Returns:
        np.ndarray of telemetry values (n_timesteps, n_features).
    """
    path = os.path.join(DATA_DIR, dataset, f"{split}", f"{channel}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Channel data not found: {path}")
    return np.load(path)


def load_smap_multi_channel(max_channels=25):
    """Load multiple SMAP channels for multi-sensor anomaly detection.

    Downloads SMAP data if not present, loads channels from different
    sensor types (temperature, power, ground systems, etc.), extracts
    primary telemetry values, and combines into multi-channel arrays.

    Args:
        max_channels: Maximum number of channels to load.

    Returns:
        train_data: (n_timesteps, n_channels) training array.
        test_data: (n_timesteps, n_channels) test array.
        test_labels: (n_timesteps,) binary anomaly labels.
        channel_names: List of channel name strings.
        channel_types: List of sensor type strings per channel.
    """
    # Download SMAP dataset and labels
    download_dataset("SMAP")
    labels_path = download_labels()
    anomaly_info = parse_anomaly_labels(labels_path, "SMAP")

    smap_dir = os.path.join(DATA_DIR, "SMAP")
    train_dir = os.path.join(smap_dir, "train")
    test_dir = os.path.join(smap_dir, "test")

    # Get available channels that exist in both train and test
    available_train = {f.replace(".npy", "") for f in os.listdir(train_dir) if f.endswith(".npy")}
    available_test = {f.replace(".npy", "") for f in os.listdir(test_dir) if f.endswith(".npy")}
    available = available_train & available_test

    # Select channels by type for balanced multi-sensor representation
    selected = []
    selected_types = []
    for sensor_type, channels in SMAP_CHANNELS.items():
        for ch in channels:
            if ch in available and len(selected) < max_channels:
                selected.append(ch)
                selected_types.append(sensor_type)

    if not selected:
        raise RuntimeError("No SMAP channels found. Check data download.")

    print(f"\nLoading {len(selected)} SMAP channels: {selected}")

    # Load primary telemetry signal (column 0) from each channel
    train_signals = []
    test_signals = []
    final_channels = []
    final_types = []

    for ch, ch_type in zip(selected, selected_types):
        try:
            train_ch = load_channel("SMAP", ch, "train")[:, 0]
            test_ch = load_channel("SMAP", ch, "test")[:, 0]
            train_signals.append(train_ch)
            test_signals.append(test_ch)
            final_channels.append(ch)
            final_types.append(ch_type)
        except (FileNotFoundError, IndexError) as e:
            print(f"  Skipping {ch}: {e}")
            continue

    # Truncate to minimum length across channels
    min_train = min(len(s) for s in train_signals)
    min_test = min(len(s) for s in test_signals)

    train_data = np.column_stack([s[:min_train] for s in train_signals])
    test_data = np.column_stack([s[:min_test] for s in test_signals])

    # Create anomaly labels (union across all selected channels)
    test_labels = np.zeros(min_test, dtype=int)
    for ch_name in final_channels:
        if ch_name in anomaly_info:
            for start, end in anomaly_info[ch_name]:
                s = max(0, min(start, min_test - 1))
                e = min(end, min_test)
                test_labels[s:e] = 1

    anomaly_pct = test_labels.mean() * 100
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape:  {test_data.shape}")
    print(f"Anomaly ratio: {anomaly_pct:.1f}% ({test_labels.sum()} / {len(test_labels)} timesteps)")
    print(f"Channel types: {dict(zip(final_channels, final_types))}")

    return train_data, test_data, test_labels, final_channels, final_types


def load_smap_single_entity(channel="T-1"):
    """Load a single SMAP entity with all 25 features.

    Each SMAP entity has shape (timesteps, 25) — 1 telemetry value
    plus 24 telecommand features. This uses the full feature set
    for one channel.

    Args:
        channel: SMAP channel name (e.g., 'T-1').

    Returns:
        train_data, test_data, test_labels.
    """
    download_dataset("SMAP")
    labels_path = download_labels()
    anomaly_info = parse_anomaly_labels(labels_path, "SMAP")

    train_data = load_channel("SMAP", channel, "train")
    test_data = load_channel("SMAP", channel, "test")

    n_test = len(test_data)
    test_labels = np.zeros(n_test, dtype=int)
    if channel in anomaly_info:
        for start, end in anomaly_info[channel]:
            s = max(0, min(start, n_test - 1))
            e = min(end, n_test)
            test_labels[s:e] = 1

    print(f"Channel {channel}: train={train_data.shape}, test={test_data.shape}")
    print(f"Anomaly ratio: {test_labels.mean() * 100:.1f}%")

    return train_data, test_data, test_labels


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
    parser.add_argument("--dataset", choices=["SMAP", "MSL", "both", "synthetic", "smap-multi"],
                        default="smap-multi", help="Dataset to download/generate")
    parser.add_argument("--max-channels", type=int, default=25)
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

    elif args.dataset == "smap-multi":
        print("Loading NASA SMAP multi-channel data...")
        train, test, labels, ch_names, ch_types = load_smap_multi_channel(
            max_channels=args.max_channels
        )
        np.save(os.path.join(DATA_DIR, "train_data.npy"), train)
        np.save(os.path.join(DATA_DIR, "test_data.npy"), test)
        np.save(os.path.join(DATA_DIR, "test_labels.npy"), labels)
        # Save channel metadata
        np.save(os.path.join(DATA_DIR, "channel_names.npy"), np.array(ch_names))
        np.save(os.path.join(DATA_DIR, "channel_types.npy"), np.array(ch_types))
        print(f"\nData saved to {DATA_DIR}")

    else:
        if args.dataset in ("SMAP", "both"):
            download_dataset("SMAP")
        if args.dataset in ("MSL", "both"):
            download_dataset("MSL")
        download_labels()
