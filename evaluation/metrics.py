"""Classification and probabilistic metrics for HAR evaluation."""

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from config import STATE_NAMES


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    state_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute rigorous classification metrics: accuracy, per-class P/R/F1, macro & weighted averages.

    Args:
        y_true: Ground truth state labels, 1D array of integers.
        y_pred: Predicted state labels, 1D array of integers.
        state_names: List of class names corresponding to state indices.

    Returns:
        Dictionary of metrics with raw and formatted values.
    """
    y_t = np.asarray(y_true, dtype=np.int32).ravel()
    y_p = np.asarray(y_pred, dtype=np.int32).ravel()

    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty label arrays.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    names = state_names or list(STATE_NAMES)
    labels = list(range(len(names)))

    accuracy = float(sk_metrics.accuracy_score(y_t, y_p))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        sk_metrics.precision_recall_fscore_support(y_t, y_p, labels=labels, zero_division=0)
    )

    macro_f1 = float(sk_metrics.f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(sk_metrics.f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Confusion matrix: raw counts and row-normalized (recall)
    cm_raw = sk_metrics.confusion_matrix(y_t, y_p, labels=labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw.astype(float),
        row_sums,
        out=np.zeros_like(cm_raw, dtype=float),
        where=row_sums != 0,
    )

    per_class_results = {}
    for idx, name in enumerate(names):
        per_class_results[name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": int(support_per_class[idx]),
        }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_results,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "state_names": names,
        "total_samples": len(y_t),
    }


def compute_sequence_log_likelihood_stats(
    log_likelihoods: list[float], sequence_lengths: list[int]
) -> dict[str, float]:
    """Compute per-sequence and per-time-step log-likelihood summary statistics."""
    if not log_likelihoods:
        return {"mean_ll": 0.0, "std_ll": 0.0, "mean_nll_per_step": 0.0}

    ll_arr = np.array(log_likelihoods, dtype=np.float64)
    lengths_arr = np.array(sequence_lengths, dtype=np.float64)

    per_step_nll = -ll_arr / np.maximum(1.0, lengths_arr)

    return {
        "mean_log_likelihood": float(np.mean(ll_arr)),
        "std_log_likelihood": float(np.std(ll_arr)),
        "mean_nll_per_step": float(np.mean(per_step_nll)),
        "median_log_likelihood": float(np.median(ll_arr)),
    }
