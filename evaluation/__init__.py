"""Evaluation modules for HAR HMM performance benchmarking."""

from .bootstrap import compute_bootstrap_confidence_intervals
from .calibration import compute_reliability_curve
from .error_analysis import analyze_locomotion_errors
from .loso import run_loso_cross_validation
from .metrics import compute_classification_metrics, compute_sequence_log_likelihood_stats

__all__ = [
    "compute_classification_metrics",
    "compute_sequence_log_likelihood_stats",
    "compute_reliability_curve",
    "compute_bootstrap_confidence_intervals",
    "run_loso_cross_validation",
    "analyze_locomotion_errors",
]

