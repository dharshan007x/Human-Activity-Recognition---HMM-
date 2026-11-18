"""Loader for PAMAP2 dataset files with automatic fallback to synthetic sample."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CHEST_ACCEL_16G_COLS,
    CHEST_ACCEL_NAMES,
    DECIMATION_FACTOR,
    RAW_DATA_DIR,
    SYNTHETIC_SAMPLE_PATH,
    TARGET_ACTIVITIES,
)
from preprocessing.clean import clean_accelerometer_data, interpolate_short_gaps
from preprocessing.features import extract_features
from preprocessing.filter import decimate_signal


def load_pamap2_raw_file(file_path: Path | str, subject_id: int) -> pd.DataFrame:
    """Parse raw 54-column PAMAP2 .dat whitespace-delimited file.

    Extracts:
    - Col 0: timestamp (s)
    - Col 1: activity_id
    - Cols 21, 22, 23: chest accelerometer +-16g (m/s^2)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw PAMAP2 file not found: {path}")

    # Read selected columns only for memory efficiency
    usecols = [0, 1] + CHEST_ACCEL_16G_COLS
    names = ["timestamp", "activity_id"] + CHEST_ACCEL_NAMES

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=usecols,
        names=names,
        dtype=np.float64,
        engine="c",
    )

    df["subject_id"] = int(subject_id)
    # Convert activity_id to integer
    df["activity_id"] = df["activity_id"].fillna(0).astype(int)

    # Filter out transient activities (activity_id 0) and non-target activities
    df = df[df["activity_id"].isin(TARGET_ACTIVITIES.keys())].copy()
    df["activity_name"] = df["activity_id"].map(TARGET_ACTIVITIES)

    # Sort strictly by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_pamap2_subject(
    subject_id: int,
    raw_dir: Path = RAW_DATA_DIR,
    decimate: bool = True,
) -> pd.DataFrame:
    """Load and preprocess a single subject's PAMAP2 recording.

    Applies:
    1. Physical outlier check (|a| <= 16g).
    2. Short-gap interpolation strictly within activity segments.
    3. 4th-order zero-phase Butterworth anti-alias filter + decimation (100 -> 25 Hz).
    4. Euclidean magnitude feature extraction.
    """
    # Possible locations for subject file
    candidate_paths = [
        raw_dir / f"subject{subject_id}.dat",
        raw_dir / "PAMAP2_Dataset" / "Protocol" / f"subject{subject_id}.dat",
        raw_dir / "Protocol" / f"subject{subject_id}.dat",
    ]

    target_file = None
    for p in candidate_paths:
        if p.exists():
            target_file = p
            break

    if target_file is None:
        raise FileNotFoundError(
            f"Could not locate subject file for subject {subject_id} in {raw_dir}"
        )

    df_raw = load_pamap2_raw_file(target_file, subject_id)
    df_clean = clean_accelerometer_data(df_raw, accel_cols=CHEST_ACCEL_NAMES)
    df_interp = interpolate_short_gaps(df_clean, accel_cols=CHEST_ACCEL_NAMES)

    if decimate and DECIMATION_FACTOR > 1:
        # Decimate segment-by-segment to prevent discontinuous boundary transients
        decimated_segments = []
        for act_id, group in df_interp.groupby("activity_id", sort=False):
            if len(group) < 30:
                continue

            acc_matrix = group[CHEST_ACCEL_NAMES].to_numpy()
            acc_decimated = decimate_signal(acc_matrix, factor=DECIMATION_FACTOR, apply_filter=True)

            timestamps = group["timestamp"].to_numpy()[::DECIMATION_FACTOR]
            n_rows = len(acc_decimated)

            seg_df = pd.DataFrame(
                {
                    "timestamp": timestamps[:n_rows],
                    "subject_id": subject_id,
                    "activity_id": act_id,
                    "activity_name": TARGET_ACTIVITIES[act_id],
                    "acc_x": acc_decimated[:, 0],
                    "acc_y": acc_decimated[:, 1],
                    "acc_z": acc_decimated[:, 2],
                }
            )
            decimated_segments.append(seg_df)

        if decimated_segments:
            df_final = pd.concat(decimated_segments, ignore_index=True)
        else:
            df_final = df_interp
    else:
        df_final = df_interp

    df_feat = extract_features(df_final)
    return df_feat


def load_processed_or_synthetic(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, bool]:
    """Load PAMAP2 real subjects if present; otherwise load the synthetic fallback.

    Returns:
        (df, is_synthetic_flag)
    """
    available_subjects = []
    # Search for protocol files
    for sid in range(101, 110):
        for candidate in [
            raw_dir / f"subject{sid}.dat",
            raw_dir / "PAMAP2_Dataset" / "Protocol" / f"subject{sid}.dat",
            raw_dir / "Protocol" / f"subject{sid}.dat",
        ]:
            if candidate.exists():
                available_subjects.append(sid)
                break

    if available_subjects:
        print(f"[INFO] Found {len(available_subjects)} raw PAMAP2 subjects: {available_subjects}")
        dfs = [load_pamap2_subject(sid, raw_dir=raw_dir) for sid in available_subjects]
        combined = pd.concat(dfs, ignore_index=True)
        return combined, False
    else:
        print(f"[INFO] Raw PAMAP2 files not found in {raw_dir}.")
        print(f"[INFO] Loading validated synthetic demo dataset: {SYNTHETIC_SAMPLE_PATH}")
        if not SYNTHETIC_SAMPLE_PATH.exists():
            from data.generate_synthetic_sample import generate_synthetic_data

            df_syn = generate_synthetic_data()
            df_syn.to_csv(SYNTHETIC_SAMPLE_PATH, index=False)
        else:
            df_syn = pd.read_csv(SYNTHETIC_SAMPLE_PATH)

        df_feat = extract_features(df_syn)
        return df_feat, True
