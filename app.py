"""Gradio demo for satellite telemetry anomaly detection.

Interactive web interface for visualizing anomaly detection results
on NASA SMAP satellite telemetry data.

Usage:
    python app.py
"""

import os
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from model import LSTMAutoencoder
from train import create_windows, normalize_data
from detect import (
    compute_reconstruction_errors,
    compute_per_channel_errors,
    compute_threshold,
    compute_adaptive_threshold,
    detect_anomalies,
    smooth_errors,
    rank_anomalous_channels,
)


# Load config and model at startup
CONFIG_PATH = "configs/config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = CONFIG["data"]["data_dir"]
MODEL_DIR = CONFIG["output"]["model_dir"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load data
train_data = np.load(os.path.join(DATA_DIR, "train_data.npy"))
test_data = np.load(os.path.join(DATA_DIR, "test_data.npy"))
test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

channel_names_path = os.path.join(DATA_DIR, "channel_names.npy")
if os.path.exists(channel_names_path):
    CHANNEL_NAMES = list(np.load(channel_names_path, allow_pickle=True))
else:
    CHANNEL_NAMES = [f"ch_{i}" for i in range(train_data.shape[1])]

channel_types_path = os.path.join(DATA_DIR, "channel_types.npy")
if os.path.exists(channel_types_path):
    CHANNEL_TYPES = list(np.load(channel_types_path, allow_pickle=True))
else:
    CHANNEL_TYPES = ["sensor"] * train_data.shape[1]

# Update config with actual dims
CONFIG["model"]["input_dim"] = train_data.shape[1]

# Normalize
train_norm, test_norm, _ = normalize_data(train_data, test_data)

# Create windows
window_size = CONFIG["data"]["window_size"]
train_windows = create_windows(train_norm, window_size)
test_windows = create_windows(test_norm, window_size)

# Load model
model_path = os.path.join(MODEL_DIR, "best_model.pt")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

MODEL = LSTMAutoencoder(
    input_dim=CONFIG["model"]["input_dim"],
    hidden_dim=CONFIG["model"]["hidden_dim"],
    latent_dim=CONFIG["model"]["latent_dim"],
    seq_len=window_size,
    num_layers=CONFIG["model"]["num_layers"],
    dropout=CONFIG["model"]["dropout"],
).to(DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()

# Precompute errors
TRAIN_ERRORS = compute_reconstruction_errors(MODEL, train_windows, DEVICE)
TEST_ERRORS = compute_reconstruction_errors(MODEL, test_windows, DEVICE)
CHANNEL_ERRORS = compute_per_channel_errors(MODEL, test_windows, DEVICE)
TEST_ERRORS_SMOOTH = smooth_errors(TEST_ERRORS, CONFIG["detection"]["smoothing_window"])


def run_detection(sigma, use_adaptive, adaptive_window, min_length, start_idx, end_idx):
    """Run anomaly detection with user-specified parameters."""
    # Compute threshold
    if use_adaptive:
        threshold = compute_adaptive_threshold(
            TEST_ERRORS_SMOOTH, window_size=int(adaptive_window), sigma=sigma
        )
    else:
        threshold = compute_threshold(TRAIN_ERRORS, sigma)

    # Detect anomalies
    predictions, segments = detect_anomalies(
        TEST_ERRORS_SMOOTH, threshold, min_length=int(min_length)
    )

    # Slice to view range
    s = int(start_idx)
    e = min(int(end_idx), len(TEST_ERRORS_SMOOTH))

    # --- Plot 1: Reconstruction error timeline ---
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    timesteps = np.arange(s, e)
    ax1.plot(timesteps, TEST_ERRORS_SMOOTH[s:e], "k-", linewidth=0.6, alpha=0.8, label="Error (smoothed)")

    if isinstance(threshold, np.ndarray):
        ax1.plot(timesteps, threshold[s:e], "r--", linewidth=1.2, label="Adaptive Threshold")
    else:
        ax1.axhline(y=threshold, color="r", linestyle="--", linewidth=1.2, label=f"Threshold (σ={sigma})")

    anomaly_mask = predictions[s:e].astype(bool)
    ax1.fill_between(timesteps, 0, TEST_ERRORS_SMOOTH[s:e],
                     where=anomaly_mask, color="red", alpha=0.3, label="Detected Anomaly")

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Reconstruction Error (MSE)")
    ax1.set_title("Anomaly Detection — Reconstruction Error Timeline")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Plot 2: Multi-channel view with anomaly highlighting ---
    n_show = min(6, len(CHANNEL_NAMES))
    fig2, axes = plt.subplots(n_show + 1, 1, figsize=(14, 2.5 * (n_show + 1)), sharex=True)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    anom_mask_full = np.zeros(len(test_data), dtype=bool)
    anom_mask_full[window_size - 1 : window_size - 1 + len(predictions)] = predictions.astype(bool)

    for i in range(n_show):
        ax = axes[i]
        ch_data = test_data[s:e, i]
        ax.plot(np.arange(s, e), ch_data, color=colors[i % len(colors)], linewidth=0.6, alpha=0.8)
        ax.fill_between(np.arange(s, e), ch_data.min(), ch_data.max(),
                        where=anom_mask_full[s:e], color="red", alpha=0.15)
        ch_label = f"{CHANNEL_NAMES[i]} ({CHANNEL_TYPES[i]})" if i < len(CHANNEL_TYPES) else CHANNEL_NAMES[i]
        ax.set_ylabel(ch_label, fontsize=8)
        ax.grid(True, alpha=0.2)

    # Bottom: ground truth vs predictions
    ax = axes[-1]
    aligned_labels = test_labels[s:e]
    ax.fill_between(np.arange(s, e), 0, aligned_labels,
                    alpha=0.3, color="blue", label="Ground Truth")
    ax.fill_between(np.arange(s, e), 0, anom_mask_full[s:e].astype(float) * 0.8,
                    alpha=0.3, color="red", label="Predicted")
    ax.set_ylabel("Anomaly")
    ax.set_xlabel("Timestep")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Multi-Channel Satellite Telemetry — Anomaly Detection", fontsize=12, y=1.01)
    plt.tight_layout()

    # --- Plot 3: Per-channel anomaly scores ---
    fig3, ax3 = plt.subplots(figsize=(14, 4))

    # Average channel error in anomalous vs normal regions
    normal_mask = predictions == 0
    anom_mask_pred = predictions == 1

    if anom_mask_pred.sum() > 0 and normal_mask.sum() > 0:
        normal_ch_err = CHANNEL_ERRORS[normal_mask].mean(axis=0)
        anom_ch_err = CHANNEL_ERRORS[anom_mask_pred].mean(axis=0)
        anomaly_scores = anom_ch_err / (normal_ch_err + 1e-8)

        sorted_idx = np.argsort(anomaly_scores)[::-1]
        top_n = min(15, len(sorted_idx))
        bar_names = [CHANNEL_NAMES[i] for i in sorted_idx[:top_n]]
        bar_scores = anomaly_scores[sorted_idx[:top_n]]

        bars = ax3.barh(range(top_n), bar_scores, color=["#F44336" if s > 2 else "#FF9800" if s > 1.5 else "#4CAF50" for s in bar_scores])
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(bar_names, fontsize=9)
        ax3.set_xlabel("Anomaly Score (× baseline error)")
        ax3.set_title("Per-Channel Anomaly Scoring — Root Cause Analysis")
        ax3.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0×)")
        ax3.legend()
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, "No anomalies detected with current settings",
                ha="center", va="center", transform=ax3.transAxes)

    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    # Summary text
    n_anomalous = int(predictions.sum())
    n_total = len(predictions)
    summary = f"**Detection Results:**\n"
    summary += f"- Detected **{len(segments)} anomalous segments** ({n_anomalous}/{n_total} points, {n_anomalous / n_total * 100:.1f}%)\n"
    summary += f"- Threshold: {'Adaptive' if use_adaptive else 'Static'} (σ={sigma})\n"
    summary += f"- Channels: {len(CHANNEL_NAMES)} sensors\n\n"

    if segments:
        summary += "**Top Anomaly Segments:**\n"
        for i, (seg_s, seg_e) in enumerate(segments[:5]):
            summary += f"- Segment {i+1}: timesteps {seg_s}–{seg_e} (length={seg_e - seg_s})\n"

    return fig1, fig2, fig3, summary


# Build Gradio interface
with gr.Blocks(title="Satellite Telemetry Anomaly Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🛰️ Satellite Telemetry Anomaly Detection\n"
        "LSTM-Autoencoder for detecting anomalies in NASA SMAP satellite telemetry. "
        "Adjust detection parameters and explore multi-channel sensor data with "
        "per-channel anomaly scoring for root cause analysis."
    )

    with gr.Row():
        with gr.Column(scale=1):
            sigma = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Threshold Sigma (σ)")
            use_adaptive = gr.Checkbox(value=False, label="Use Adaptive Threshold")
            adaptive_window = gr.Slider(50, 500, value=200, step=10,
                                       label="Adaptive Window Size")
            min_length = gr.Slider(1, 20, value=3, step=1,
                                   label="Min Anomaly Length")
            start_idx = gr.Slider(0, len(test_data) - 100, value=0, step=100,
                                  label="View Start Index")
            end_idx = gr.Slider(100, len(test_data), value=min(2000, len(test_data)),
                                step=100, label="View End Index")
            run_btn = gr.Button("Run Detection", variant="primary")

        with gr.Column(scale=3):
            summary_out = gr.Markdown()

    with gr.Tabs():
        with gr.TabItem("Reconstruction Error"):
            plot1 = gr.Plot(label="Reconstruction Error Timeline")
        with gr.TabItem("Multi-Channel View"):
            plot2 = gr.Plot(label="Multi-Channel Telemetry")
        with gr.TabItem("Per-Channel Scoring"):
            plot3 = gr.Plot(label="Per-Channel Anomaly Scores")

    run_btn.click(
        fn=run_detection,
        inputs=[sigma, use_adaptive, adaptive_window, min_length, start_idx, end_idx],
        outputs=[plot1, plot2, plot3, summary_out],
    )

    # Run initial detection on load
    demo.load(
        fn=lambda: run_detection(3.0, False, 200, 3, 0, min(2000, len(test_data))),
        outputs=[plot1, plot2, plot3, summary_out],
    )


if __name__ == "__main__":
    demo.launch(share=False)
