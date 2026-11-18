"""Leave-One-Subject-Out (LOSO) Cross-Validation engine."""

from typing import Any

import numpy as np

from config import STATE_NAMES
from evaluation.metrics import compute_classification_metrics
from models.hmm_model import GaussianHMM
from preprocessing.split import SubjectScaler, SubjectSplitter


def run_loso_cross_validation(
    sequences: list[np.ndarray],
    labels: list[int],
    subject_ids: list[int],
    state_names: list[str] | None = None,
) -> dict[str, Any]:
    """Execute complete Leave-One-Subject-Out Cross-Validation.

    ZERO LEAKAGE:
    For every fold, the SubjectScaler and GaussianHMM parameters are fit
    strictly on the K-1 training participants.

    Args:
        sequences: List of arrays of shape (T, D).
        labels: List of integer state labels per sequence.
        subject_ids: Participant ID for each sequence.
        state_names: List of state names.

    Returns:
        Dictionary of per-fold results, fold details, and aggregated mean +- std.
    """
    names = state_names or list(STATE_NAMES)
    splitter = SubjectSplitter()

    folds = list(splitter.loso_folds(sequences, labels, subject_ids))
    if not folds:
        return {"error": "Not enough subjects to perform LOSO cross-validation."}

    fold_results = []
    fold_accuracies = []
    fold_macro_f1s = []

    for held_out_subject, (train_x, train_y), (test_x, test_y) in folds:
        # Fit scaler exclusively on training subjects of this fold
        scaler = SubjectScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        # Build state sequences for supervised fit
        train_states = [np.full(len(x), y, dtype=np.int32) for x, y in zip(train_x, train_y)]

        # Train HMM model
        hmm_model = GaussianHMM(
            n_states=len(names), n_features=train_x[0].shape[1], state_names=names
        )
        hmm_model.fit(train_x_scaled, train_states)

        # Batch evaluate on held-out subject
        all_true = []
        all_pred = []

        for x_seq, y_true_int in zip(test_x_scaled, test_y):
            # Pointwise posterior prediction
            y_pred = hmm_model.predict(x_seq)
            all_true.extend([y_true_int] * len(x_seq))
            all_pred.extend(y_pred.tolist())

        metrics = compute_classification_metrics(all_true, all_pred, state_names=names)
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]

        fold_accuracies.append(acc)
        fold_macro_f1s.append(f1)

        fold_results.append(
            {
                "held_out_subject": int(held_out_subject),
                "n_train_sequences": len(train_x),
                "n_test_sequences": len(test_x),
                "accuracy": acc,
                "macro_f1": f1,
                "per_class": metrics["per_class"],
            }
        )

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_f1 = float(np.mean(fold_macro_f1s))
    std_f1 = float(np.std(fold_macro_f1s))

    return {
        "n_folds": len(fold_results),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "fold_results": fold_results,
    }
