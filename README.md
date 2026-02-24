# Satellite Telemetry Anomaly Detection

An **LSTM-Autoencoder** for detecting anomalies in satellite sensor telemetry trained on NASA **SMAP** (Soil Moisture Active Passive) satellite data. Features per-channel anomaly scoring, adaptive confidence bands, and interactive Gradio visualization.

## Overview

Satellite telemetry data often contains subtle anomalies indicating sensor malfunctions, attitude control issues, or environmental events. This project uses an LSTM-Autoencoder that learns to reconstruct normal operational patterns — deviations beyond a learned threshold are flagged as anomalies.

## Architecture

```
Input Window (50 × N) → LSTM Encoder (2 layers, 64 hidden)
    → Latent Space (32-dim) → LSTM Decoder → Reconstructed Window
    → Reconstruction Error (MSE) → Threshold Detection → Anomaly Labels
    → Per-Channel Scoring → Root Cause Analysis
```

## Results (NASA SMAP Data)

| Metric | Score |
|--------|-------|
| Point-Adjust F1 | **0.964** |
| Point-Based F1 | 0.906 |
| Precision | 0.923 |
| Recall | 0.890 |
| ROC AUC | 0.974 |
| Accuracy | 93.0% |

## Features

- **Real NASA SMAP data** — downloads and processes SMAP satellite telemetry from Kaggle
- **Per-channel anomaly scoring** — identifies which sensors contribute most to detected anomalies for root cause analysis
- **Adaptive confidence bands** — dynamic threshold that adjusts to local error statistics via rolling window
- **Point-adjust evaluation** — standard metric for time series anomaly detection (Xu et al., WWW 2018)
- **Interactive Gradio demo** — explore detection results with adjustable parameters
- **Multi-channel visualization** — time-series overlay plots with anomaly highlighting

## Setup

```bash
git clone https://github.com/svn05/satellite-anomaly-detection.git
cd satellite-anomaly-detection
pip install -r requirements.txt
```

## Usage

### 1. Prepare data
```bash
# Download NASA SMAP data (requires Kaggle account)
python data/download_data.py --dataset smap-multi

# Or generate synthetic telemetry data (for quick testing)
python data/download_data.py --dataset synthetic
```

### 2. Train the model
```bash
python train.py
python train.py --epochs 200 --batch-size 32
```

### 3. Detect anomalies
```bash
python detect.py
python detect.py --threshold-sigma 2.5
python detect.py --adaptive --adaptive-window 200
```

### 4. Evaluate
```bash
python evaluate.py
python evaluate.py --optimize-threshold
```

### 5. Visualize (Gradio demo)
```bash
python app.py
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
├── detect.py           # Anomaly detection + per-channel scoring + adaptive thresholds
├── evaluate.py         # Point-adjust F1, precision, recall, ROC AUC
├── visualize.py        # Confidence bands + multi-channel plots
├── app.py              # Interactive Gradio demo
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
- **Gradio** — Interactive web demo
- **Matplotlib** — Confidence-band and multi-channel visualizations
- **NumPy** — Data processing and windowing

## Dataset

- **NASA SMAP** (Soil Moisture Active Passive satellite) — 55 telemetry entities with 25 features each
- Expert-labeled anomalies from real mission data
- Source: [NASA Telemanom](https://github.com/khundman/telemanom) via [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
