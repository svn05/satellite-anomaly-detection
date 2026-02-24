"""Evaluation metrics for anomaly detection.

Computes precision, recall, F1-score, and generates classification reports
using point-based, point-adjust, and segment-based evaluation.

The point-adjust approach is standard in time series anomaly detection:
if ANY predicted anomaly point falls within a true anomaly segment,
ALL points in that segment are marked as detected (true positives).

Usage:
    python evaluate.py
    python evaluate.py --optimize-threshold
"""

import argparse
import os
import numpy as np
import yaml
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def point_based_metrics(y_true, y_pred):
    """Compute point-based anomaly detection metrics.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly).
        y_pred: Predicted labels.

    Returns:
        dict with precision, recall, f1, accuracy.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": (y_true == y_pred).mean(),
    }


def point_adjust_metrics(y_true, y_pred):
    """Compute point-adjust F1 (standard for time series anomaly detection).

    If any predicted anomaly overlaps with a true anomaly segment,
    all points in that segment are counted as true positives. This
    accounts for the fact that detecting the start of an anomaly is
    sufficient for practical alerting.

    Reference: Xu et al., "Unsupervised Anomaly Detection via Variational
    Auto-Encoder for Seasonal KPIs in Web Applications" (WWW 2018)

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly).
        y_pred: Predicted labels.

    Returns:
        dict with point-adjust precision, recall, f1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Find true anomaly segments
    true_segments = _find_segments(y_true)

    # Create adjusted predictions: for each true segment that has any
    # overlap with predictions, mark the entire segment as TP
    adjusted_pred = y_pred.copy()
    for start, end in true_segments:
        if y_pred[start:end].any():
            # At least one predicted anomaly in this segment
            adjusted_pred[start:end] = 1

    return {
        "precision": precision_score(y_true, adjusted_pred, zero_division=0),
        "recall": recall_score(y_true, adjusted_pred, zero_division=0),
        "f1": f1_score(y_true, adjusted_pred, zero_division=0),
    }


def segment_based_metrics(y_true, y_pred):
    """Compute segment-based metrics.

    If any point in a true anomaly segment is detected, the entire
    segment is considered a true positive.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        dict with segment-level precision, recall, f1.
    """
    true_segments = _find_segments(y_true)
    pred_segments = _find_segments(y_pred)

    if len(true_segments) == 0 and len(pred_segments) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(true_segments) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if len(pred_segments) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    true_detected = 0
    for ts, te in true_segments:
        for ps, pe in pred_segments:
            if ps < te and pe > ts:
                true_detected += 1
                break

    pred_correct = 0
    for ps, pe in pred_segments:
        for ts, te in true_segments:
            if ps < te and pe > ts:
                pred_correct += 1
                break

    precision = pred_correct / len(pred_segments) if pred_segments else 0
    recall = true_detected / len(true_segments) if true_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def _find_segments(labels):
    """Find contiguous segments of anomalies (value=1)."""
    segments = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i
            while j < len(labels) and labels[j] == 1:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


def find_optimal_threshold(errors, labels, sigma_range=np.arange(0.5, 6.0, 0.05),
                           use_point_adjust=True):
    """Find the threshold sigma that maximizes F1-score.

    Args:
        errors: Reconstruction errors.
        labels: Ground truth labels.
        sigma_range: Range of sigma values to search.
        use_point_adjust: Use point-adjust F1 for optimization.

    Returns:
        best_sigma, best_f1, all_results.
    """
    mean = errors.mean()
    std = errors.std()

    best_sigma = 3.0
    best_f1 = 0.0
    results = []

    for sigma in sigma_range:
        threshold = mean + sigma * std
        preds = (errors > threshold).astype(int)

        if use_point_adjust:
            metrics = point_adjust_metrics(labels, preds)
            current_f1 = metrics["f1"]
        else:
            current_f1 = f1_score(labels, preds, zero_division=0)

        results.append({"sigma": float(sigma), "threshold": float(threshold), "f1": float(current_f1)})
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_sigma = float(sigma)

    return best_sigma, best_f1, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection results")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--optimize-threshold", action="store_true",
                        help="Search for optimal threshold sigma")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load results
    results_dir = config["output"]["results_dir"]
    data_dir = config["data"]["data_dir"]

    predictions = np.load(os.path.join(results_dir, "predictions.npy"))
    test_errors = np.load(os.path.join(results_dir, "test_errors.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Align labels with windowed predictions
    window_size = config["data"]["window_size"]
    labels = test_labels[window_size - 1:]
    labels = labels[:len(predictions)]

    print("=" * 60)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 60)

    # Point-based metrics
    point_metrics = point_based_metrics(labels, predictions)
    print("\n--- Point-Based Metrics ---")
    print(f"  Precision: {point_metrics['precision']:.4f}")
    print(f"  Recall:    {point_metrics['recall']:.4f}")
    print(f"  F1-Score:  {point_metrics['f1']:.4f}")
    print(f"  Accuracy:  {point_metrics['accuracy']:.4f}")

    # Point-adjust metrics (standard for time series AD)
    pa_metrics = point_adjust_metrics(labels, predictions)
    print("\n--- Point-Adjust Metrics (Standard for Time Series AD) ---")
    print(f"  Precision: {pa_metrics['precision']:.4f}")
    print(f"  Recall:    {pa_metrics['recall']:.4f}")
    print(f"  F1-Score:  {pa_metrics['f1']:.4f}")

    # Segment-based metrics
    seg_metrics = segment_based_metrics(labels, predictions)
    print("\n--- Segment-Based Metrics ---")
    print(f"  Precision: {seg_metrics['precision']:.4f}")
    print(f"  Recall:    {seg_metrics['recall']:.4f}")
    print(f"  F1-Score:  {seg_metrics['f1']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\n--- Confusion Matrix ---")
    print(f"  TN={cm[0, 0]:>6d}  FP={cm[0, 1]:>6d}")
    print(f"  FN={cm[1, 0]:>6d}  TP={cm[1, 1]:>6d}")

    # Classification report
    print("\n--- Full Classification Report ---")
    print(classification_report(labels, predictions, target_names=["Normal", "Anomaly"]))

    # ROC AUC
    try:
        auc = roc_auc_score(labels, test_errors[:len(labels)])
        print(f"ROC AUC Score: {auc:.4f}")
    except ValueError:
        print("ROC AUC: N/A (single class in labels)")

    # Optimal threshold search
    if args.optimize_threshold:
        print("\n--- Threshold Optimization (Point-Adjust F1) ---")
        best_sigma, best_f1, all_results = find_optimal_threshold(
            test_errors[:len(labels)], labels, use_point_adjust=True
        )
        print(f"  Optimal sigma: {best_sigma:.2f}")
        print(f"  Best Point-Adjust F1: {best_f1:.4f}")

        # Show top 5 results
        sorted_results = sorted(all_results, key=lambda x: -x["f1"])
        for r in sorted_results[:5]:
            print(f"    sigma={r['sigma']:.2f}: F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
