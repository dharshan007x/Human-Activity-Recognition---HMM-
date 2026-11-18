"""Data cleaning, gap interpolation, and physical outlier rejection."""

import numpy as np
import pandas as pd

from config import MAX_PHYSICAL_ACCEL_MS2


def clean_accelerometer_data(
    df: pd.DataFrame,
    accel_cols: list[str] | None = None,
    max_physical_limit: float = MAX_PHYSICAL_ACCEL_MS2,
) -> pd.DataFrame:
    """Validate physical plausibility of accelerometer values.

    Removes or flags hardware sensor anomalies where |a| > 16g (156.96 m/s^2).
    """
    if accel_cols is None:
        accel_cols = ["acc_x", "acc_y", "acc_z"]

    df_clean = df.copy()

    for col in accel_cols:
        if col in df_clean.columns:
            # Detect unphysical values exceeding +-16g limit
            unphysical_mask = df_clean[col].abs() > max_physical_limit
            if unphysical_mask.any():
                df_clean.loc[unphysical_mask, col] = np.nan

    return df_clean


def interpolate_short_gaps(
    df: pd.DataFrame,
    accel_cols: list[str] | None = None,
    max_gap_samples: int = 12,
) -> pd.DataFrame:
    """Perform time-aware linear interpolation on short missing bursts.

    CRITICAL LEAKAGE RESTRICTION:
    Interpolation is strictly applied within the grouped partitions of:
    (subject_id, activity_id).
    It is mathematically forbidden to interpolate across subject boundaries
    or across activity transitions.

    Args:
        df: DataFrame containing accelerometer data and subject/activity keys.
        accel_cols: List of accelerometer column names.
        max_gap_samples: Maximum consecutive NaNs to interpolate (e.g., 12 samples ~ 0.5s at 25 Hz).

    Returns:
        Cleaned DataFrame with short gaps interpolated and remaining long gaps dropped.
    """
    if accel_cols is None:
        accel_cols = ["acc_x", "acc_y", "acc_z"]

    df_out = df.copy()

    # Identify grouping keys
    group_keys = []
    if "subject_id" in df_out.columns:
        group_keys.append("subject_id")
    if "activity_id" in df_out.columns:
        group_keys.append("activity_id")

    if group_keys:
        grouped_dfs = []
        for _, group in df_out.groupby(group_keys, sort=False):
            g = group.copy()
            for col in accel_cols:
                if col in g.columns:
                    g[col] = g[col].interpolate(
                        method="linear", limit=max_gap_samples, limit_direction="both"
                    )
            grouped_dfs.append(g)
        if grouped_dfs:
            df_out = pd.concat(grouped_dfs, ignore_index=True)
    else:
        for col in accel_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].interpolate(
                    method="linear", limit=max_gap_samples, limit_direction="both"
                )

    # Any remaining NaNs (from long gaps > max_gap_samples) are dropped
    df_out = df_out.dropna(subset=[col for col in accel_cols if col in df_out.columns])
    return df_out.reset_index(drop=True)
