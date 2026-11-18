"""Non-parametric bootstrap estimation for 95% Confidence Intervals."""

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from config import RANDOM_SEED


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compute empirical 95% bootstrap confidence intervals for Accuracy and Macro-F1.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_bootstrap: Number of bootstrap iterations (default: 1000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing point estimates, lower CI, upper CI, and std.
    """
    rng = np.random.default_rng(seed)
    y_t = np.asarray(y_true, dtype=np.int32).ravel()
    y_p = np.asarray(y_pred, dtype=np.int32).ravel()
    n_samples = len(y_t)

    if n_samples == 0:
        return {}

    point_accuracy = float(sk_metrics.accuracy_score(y_t, y_p))
    point_macro_f1 = float(sk_metrics.f1_score(y_t, y_p, average="macro", zero_division=0))

    boot_accuracies = np.zeros(n_bootstrap, dtype=np.float64)
    boot_macro_f1s = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        b_true = y_t[indices]
        b_pred = y_p[indices]

        boot_accuracies[b] = sk_metrics.accuracy_score(b_true, b_pred)
        boot_macro_f1s[b] = sk_metrics.f1_score(b_true, b_pred, average="macro", zero_division=0)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    acc_ci_lower = float(np.percentile(boot_accuracies, lower_pct))
    acc_ci_upper = float(np.percentile(boot_accuracies, upper_pct))

    f1_ci_lower = float(np.percentile(boot_macro_f1s, lower_pct))
    f1_ci_upper = float(np.percentile(boot_macro_f1s, upper_pct))

    return {
        "n_bootstrap": n_bootstrap,
        "confidence_level": 1.0 - alpha,
        "accuracy": {
            "point_estimate": point_accuracy,
            "ci_lower": acc_ci_lower,
            "ci_upper": acc_ci_upper,
            "std": float(np.std(boot_accuracies)),
        },
        "macro_f1": {
            "point_estimate": point_macro_f1,
            "ci_lower": f1_ci_lower,
            "ci_upper": f1_ci_upper,
            "std": float(np.std(boot_macro_f1s)),
        },
    }

