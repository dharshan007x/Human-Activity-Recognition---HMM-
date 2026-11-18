"""Full training, evaluation, calibration, and LOSO benchmarking pipeline."""

import json
from pathlib import Path

import numpy as np

from config import (
    FEATURE_NAMES,
    SAVED_HMM_PATH,
    SAVED_SCALER_PATH,
    STATE_NAMES,
)
from evaluation.bootstrap import compute_bootstrap_confidence_intervals
from evaluation.calibration import compute_reliability_curve
from evaluation.error_analysis import analyze_locomotion_errors
from evaluation.loso import run_loso_cross_validation
from evaluation.metrics import (
    compute_classification_metrics,
    compute_sequence_log_likelihood_stats,
)
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler, SubjectSplitter


def run_full_pipeline(
    save_results_path: Path | str = Path(__file__).resolve().parent / "evaluation_results.json",
) -> dict:
    """Execute end-to-end data loading, subject splitting, training, and benchmarking."""
    print("[PIPELINE] 1. Loading PAMAP2 dataset or validated synthetic demo sample...")
    df, is_synthetic = load_processed_or_synthetic()
    print(f"[PIPELINE] Loaded {len(df)} rows. Is synthetic: {is_synthetic}")

    print("[PIPELINE] 2. Segmenting into contiguous temporal sequences (T=50, 25 Hz)...")
    sequences, state_sequences, labels, subject_ids = build_feature_sequences(
        df, sequence_length=50, stride=50, feature_cols=FEATURE_NAMES
    )
    print(
        f"[PIPELINE] Extracted {len(sequences)} valid sequences across subjects {sorted(list(set(subject_ids)))}."
    )

    print("[PIPELINE] 3. Partitioning sequences strictly by participant ID...")
    splitter = SubjectSplitter()
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = splitter.split(
        sequences, labels, subject_ids
    )
    print(f"[PIPELINE] Partitions: Train={len(train_x)}, Val={len(val_x)}, Test={len(test_x)}")

    print("[PIPELINE] 4. Fitting SubjectScaler strictly on Training subjects...")
    scaler = SubjectScaler()
    train_x_scaled = scaler.fit_transform(train_x, feature_names=FEATURE_NAMES)
    scaler.save(SAVED_SCALER_PATH)

    scaler.transform(val_x) if val_x else []
    test_x_scaled = scaler.transform(test_x)

    print("[PIPELINE] 5. Training GaussianHMM via supervised Maximum Likelihood Estimation...")
    train_states = [np.full(len(x), y, dtype=np.int32) for x, y in zip(train_x, train_y)]
    hmm = GaussianHMM(
        n_states=len(STATE_NAMES),
        n_features=len(FEATURE_NAMES),
        state_names=STATE_NAMES,
    )
    hmm.fit(
        train_x_scaled,
        train_states,
        metadata={
            "is_synthetic": is_synthetic,
            "train_sequences_count": len(train_x),
            "test_sequences_count": len(test_x),
            "feature_names": FEATURE_NAMES,
            "state_names": STATE_NAMES,
        },
    )
    hmm.save(SAVED_HMM_PATH)
    print(f"[PIPELINE] Model saved to {SAVED_HMM_PATH}")

    print("[PIPELINE] 6. Running batch inference on held-out test partition...")
    all_y_true = []
    all_y_pred_pointwise = []
    all_y_pred_viterbi = []
    all_confidences = []
    all_log_likelihoods = []
    seq_lengths = []

    for x_seq, y_true_int in zip(test_x_scaled, test_y):
        gamma, pointwise_states, ll, _, _ = hmm.posterior(x_seq, method="log")
        viterbi_states, _, _ = hmm.decode_viterbi(x_seq)

        max_gamma = np.max(gamma, axis=1)

        all_y_true.extend([y_true_int] * len(x_seq))
        all_y_pred_pointwise.extend(pointwise_states.tolist())
        all_y_pred_viterbi.extend(viterbi_states.tolist())
        all_confidences.extend(max_gamma.tolist())
        all_log_likelihoods.append(ll)
        seq_lengths.append(len(x_seq))

    all_y_true_arr = np.array(all_y_true, dtype=np.int32)
    all_y_pred_arr = np.array(all_y_pred_pointwise, dtype=np.int32)
    binary_accuracy = (all_y_pred_arr == all_y_true_arr).astype(int)

    print("[PIPELINE] 7. Computing test classification metrics...")
    test_metrics = compute_classification_metrics(
        all_y_true_arr, all_y_pred_arr, state_names=STATE_NAMES
    )
    viterbi_metrics = compute_classification_metrics(
        all_y_true_arr, all_y_pred_viterbi, state_names=STATE_NAMES
    )
    ll_stats = compute_sequence_log_likelihood_stats(all_log_likelihoods, seq_lengths)

    print("[PIPELINE] 8. Computing calibration and ECE...")
    calibration_results = compute_reliability_curve(all_confidences, binary_accuracy, n_bins=10)

    print("[PIPELINE] 9. Computing 95% bootstrap confidence intervals...")
    bootstrap_results = compute_bootstrap_confidence_intervals(
        all_y_true_arr, all_y_pred_arr, n_bootstrap=1000
    )

    print("[PIPELINE] 10. Running Leave-One-Subject-Out (LOSO) Cross-Validation...")
    loso_results = run_loso_cross_validation(
        sequences, labels, subject_ids, state_names=STATE_NAMES
    )

    print("[PIPELINE] 11. Running Biomechanical Error Analysis...")
    error_analysis = analyze_locomotion_errors(
        all_y_true_arr, all_y_pred_arr, test_x, state_names=STATE_NAMES
    )

    payload = {
        "dataset_info": {
            "is_synthetic": is_synthetic,
            "total_sequences": len(sequences),
            "train_sequences": len(train_x),
            "val_sequences": len(val_x),
            "test_sequences": len(test_x),
            "features": FEATURE_NAMES,
            "states": STATE_NAMES,
        },
        "model_parameters": {
            "pi": hmm.pi.tolist(),
            "A": hmm.A.tolist(),
            "state_means": [hmm.emissions[i].mean.tolist() for i in range(len(STATE_NAMES))],
            "state_covs": [hmm.emissions[i].cov.tolist() for i in range(len(STATE_NAMES))],
        },
        "test_metrics": test_metrics,
        "viterbi_metrics": viterbi_metrics,
        "likelihood_stats": ll_stats,
        "calibration": calibration_results,
        "bootstrap": bootstrap_results,
        "loso": loso_results,
        "error_analysis": error_analysis,
    }

    save_path = Path(save_results_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[PIPELINE COMPLETE] Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[PIPELINE COMPLETE] Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"[PIPELINE COMPLETE] ECE (Expected Calibration Error): {calibration_results['ece']:.4f}")
    print(f"[PIPELINE COMPLETE] LOSO Mean Accuracy: {loso_results.get('mean_accuracy', 0.0):.4f}")
    print(f"[PIPELINE COMPLETE] Saved evaluation results to {save_path}")

    return payload


if __name__ == "__main__":
    run_full_pipeline()
