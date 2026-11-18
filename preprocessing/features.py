"""Feature extraction and sequential window segmentation for HMM modeling."""

import numpy as np
import pandas as pd

from config import FEATURE_NAMES, STATE_TO_INDEX, TARGET_ACTIVITIES


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute triaxial accelerometer features and Euclidean magnitude.

    Observation vector: [acc_x, acc_y, acc_z, acc_mag]
    """
    df_feat = df.copy()

    required_axes = ["acc_x", "acc_y", "acc_z"]
    for axis in required_axes:
        if axis not in df_feat.columns:
            raise ValueError(f"Missing required accelerometer column: '{axis}'")

    # Compute Euclidean magnitude for rotational invariance
    df_feat["acc_mag"] = np.sqrt(
        df_feat["acc_x"] ** 2 + df_feat["acc_y"] ** 2 + df_feat["acc_z"] ** 2
    )

    return df_feat


def build_feature_sequences(
    df: pd.DataFrame,
    sequence_length: int = 50,
    stride: int = 50,
    feature_cols: list[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Segment continuous time-series into fixed-length contiguous sequences.

    PRESERVES TEMPORAL ORDER:
    Windows are extracted strictly in forward chronological order per subject
    and activity block. Sequences never cross subject or activity boundaries.

    Args:
        df: DataFrame containing features, activity_id, and subject_id.
        sequence_length: Number of time steps T in each sequence (e.g., 50 at 25 Hz = 2.0s).
        stride: Step size between consecutive sequences (default: non-overlapping stride = sequence_length).
        feature_cols: List of observation features (default: acc_x, acc_y, acc_z, acc_mag).

    Returns:
        sequences: List of arrays of shape (T, D).
        state_sequences: List of integer state arrays of shape (T,).
        sequence_labels: Integer label of the dominant state for each sequence.
        sequence_subjects: Subject ID corresponding to each sequence.
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES

    sequences: list[np.ndarray] = []
    state_sequences: list[np.ndarray] = []
    sequence_labels: list[int] = []
    sequence_subjects: list[int] = []

    # Filter to target activities only
    valid_df = df[df["activity_id"].isin(TARGET_ACTIVITIES.keys())].copy()

    # Group by subject and activity to prevent boundary leakage
    for (subject_id, act_id), group in valid_df.groupby(["subject_id", "activity_id"]):
        act_name = TARGET_ACTIVITIES[act_id]
        state_idx = STATE_TO_INDEX[act_name]

        features_matrix = group[feature_cols].to_numpy(dtype=np.float64)
        n_samples = len(features_matrix)

        for start_idx in range(0, n_samples - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            seq_x = features_matrix[start_idx:end_idx]
            seq_y = np.full(sequence_length, state_idx, dtype=np.int32)

            sequences.append(seq_x)
            state_sequences.append(seq_y)
            sequence_labels.append(state_idx)
            sequence_subjects.append(int(subject_id))

    return sequences, state_sequences, sequence_labels, sequence_subjects
