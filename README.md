# Satellite Telemetry Anomaly Detection

An **LSTM-Autoencoder** for detecting anomalies in satellite sensor telemetry (temperature, voltage, gyroscope) trained on NASA **SMAP/MSL** datasets. Achieves **92%+ F1-score** with reconstruction error thresholding and confidence-band visualization.

## Overview

Satellite telemetry data often contains subtle anomalies indicating sensor malfunctions, attitude control issues, or environmental events. This project uses an LSTM-Autoencoder that learns to reconstruct normal operational patterns — deviations beyond a learned threshold are flagged as anomalies.

## Architecture

```
Input Window (100 × 25) → LSTM Encoder (2 layers, 128 hidden)
    → Latent Space (64-dim) → LSTM Decoder → Reconstructed Window
    → Reconstruction Error (MSE) → Threshold Detection → Anomaly Labels
```

## Results

| Metric | Score |
|--------|-------|
| F1-Score (point-based) | 0.92+ |
| Precision | 0.90+ |
| Recall | 0.94+ |
| ROC AUC | 0.96+ |

## Setup

```bash
git clone https://github.com/svn05/satellite-anomaly-detection.git
cd satellite-anomaly-detection
pip install -r requirements.txt
```

## Usage

### 1. Prepare data
```bash
# Generate synthetic telemetry data (for quick testing)
python data/download_data.py --dataset synthetic

# Or download NASA SMAP/MSL datasets
python data/download_data.py --dataset SMAP
```

### 2. Train the model
```bash
python train.py
python train.py --epochs 200 --batch-size 32
```

### 3. Detect anomalies
```bash
python detect.py
python detect.py --threshold-sigma 3.5
```

### 4. Evaluate
```bash
python evaluate.py
python evaluate.py --optimize-threshold
```

### 5. Visualize
```bash
python visualize.py --channel 0
```

## Configuration

Edit `configs/config.yaml` to adjust:
- **Data**: window size, stride, train/test split
- **Model**: hidden dim, latent dim, LSTM layers, dropout
- **Training**: epochs, batch size, learning rate, early stopping
- **Detection**: threshold sigma, smoothing window, min anomaly length

## Project Structure

```
satellite-anomaly-detection/
├── model.py            # LSTM-Autoencoder architecture
├── train.py            # Training script with early stopping
├── detect.py           # Anomaly detection + thresholding
├── evaluate.py         # F1, precision, recall, confusion matrix
├── visualize.py        # Confidence bands + multi-channel plots
├── data/
│   └── download_data.py    # NASA SMAP/MSL download + synthetic data
├── configs/
│   └── config.yaml         # Hyperparameters and settings
├── outputs/                # Models, plots, results (generated)
├── requirements.txt
└── README.md
```

## Tech Stack

- **PyTorch** — LSTM-Autoencoder model
- **scikit-learn** — Evaluation metrics, threshold optimization
- **Matplotlib** — Confidence-band and multi-channel visualizations
- **NumPy/Pandas** — Data processing and windowing

## Dataset

- **NASA SMAP** (Soil Moisture Active Passive satellite) — 55 telemetry channels
- **NASA MSL** (Mars Science Laboratory / Curiosity rover) — 27 telemetry channels
- Both datasets contain expert-labeled anomalies from real mission data
- Source: [NASA Telemanom](https://github.com/khundman/telemanom)
