"""Reliability diagrams and Expected Calibration Error (ECE) for posterior confidence."""

from typing import Any

import numpy as np


def compute_reliability_curve(
    confidences: np.ndarray | list[float],
    accuracies: np.ndarray | list[bool | int],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve data and Expected Calibration Error (ECE).

    Args:
        confidences: Predicted maximum posterior probability per sample, in [0, 1].
        accuracies: Binary array indicating whether each prediction was correct (1) or not (0).
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Dictionary containing bin centers, bin accuracies, bin confidences, bin counts, and ECE.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    acc = np.asarray(accuracies, dtype=np.float64).ravel()

    if len(conf) == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0
    n_total = len(conf)

    bins_data = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            b_acc = float(np.mean(acc[mask]))
            b_conf = float(np.mean(conf[mask]))
            calibration_gap = abs(b_acc - b_conf)
            ece += (count / n_total) * calibration_gap
            mce = max(mce, calibration_gap)
        else:
            b_acc = 0.0
            b_conf = float(bin_centers[i])
            calibration_gap = 0.0

        bin_accs.append(b_acc)
        bin_confs.append(b_conf)

        bins_data.append(
            {
                "bin_index": i,
                "bin_lower": float(low),
                "bin_upper": float(high),
                "bin_center": float(bin_centers[i]),
                "accuracy": b_acc,
                "confidence": b_conf,
                "count": count,
                "weight": count / n_total,
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bins": bins_data,
        "total_samples": n_total,
    }
