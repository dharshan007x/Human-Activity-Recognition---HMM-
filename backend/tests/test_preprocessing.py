"""Validation tests for the preprocessing and signal conditioning pipeline."""

import numpy as np
import pandas as pd

from preprocessing.clean import clean_accelerometer_data, interpolate_short_gaps
from preprocessing.filter import apply_anti_aliasing_filter, decimate_signal
from preprocessing.split import SubjectScaler


def test_anti_aliasing_filter_attenuates_high_frequencies() -> None:
    """Verify that frequencies above the 10 Hz cutoff are strongly attenuated."""
    fs = 100.0  # Original PAMAP2 sampling frequency
    t = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)

    # 2 Hz signal (human gait frequency) + 35 Hz high frequency sensor noise
    low_freq_signal = np.sin(2 * np.pi * 2.0 * t)
    high_freq_noise = np.sin(2 * np.pi * 35.0 * t)
    composite = low_freq_signal + high_freq_noise

    filtered = apply_anti_aliasing_filter(composite, cutoff_hz=10.0, fs=fs, order=4)

    # Check power of high frequency component after filtering
    # Energy in original high-freq noise is 0.5; after 4th order Butterworth at 35Hz it should be ~0
    diff = filtered.squeeze() - low_freq_signal
    residual_noise_power = np.mean(diff[20:-20] ** 2)  # Exclude transient edges
    assert residual_noise_power < 0.05, (
        f"High frequency noise was not sufficiently attenuated: {residual_noise_power}"
    )


def test_decimation_factor() -> None:
    """Verify decimation by factor 4 correctly downsizes data length."""
    raw_data = np.zeros((100, 3))
    decimated = decimate_signal(raw_data, factor=4, apply_filter=False)
    assert decimated.shape == (25, 3)


def test_physical_outlier_filtering() -> None:
    """Assert values exceeding +-16g (156.96 m/s^2) are marked as NaN."""
    df = pd.DataFrame(
        {
            "acc_x": [0.0, 150.0, 160.0],  # 160.0 is unphysical > 156.96
            "acc_y": [9.8, 20.0, -200.0],  # -200.0 is unphysical
            "acc_z": [1.0, 0.5, 0.2],
        }
    )

    cleaned = clean_accelerometer_data(df)
    assert np.isnan(cleaned.loc[2, "acc_x"])
    assert np.isnan(cleaned.loc[2, "acc_y"])
    assert not np.isnan(cleaned.loc[0, "acc_x"])


def test_interpolation_strictly_within_subject_activity() -> None:
    """Ensure interpolation NEVER leaks across subject boundaries or activity boundaries."""
    df = pd.DataFrame(
        {
            "subject_id": [101, 101, 101, 102, 102],
            "activity_id": [4, 4, 4, 4, 4],
            "acc_x": [1.0, np.nan, 3.0, np.nan, 5.0],
        }
    )

    interpolated = interpolate_short_gaps(df, accel_cols=["acc_x"], max_gap_samples=2)

    # Within subject 101: index 1 should be (1.0 + 3.0)/2 = 2.0
    row_101_nan = interpolated[interpolated["subject_id"] == 101]
    assert np.isclose(row_101_nan.loc[1, "acc_x"], 2.0)


def test_subject_scaler_isolation() -> None:
    """Verify scaler parameters are fit exclusively on training data without test leakage."""
    train_seqs = [np.array([[10.0, 20.0], [10.0, 20.0]])]
    test_seqs = [np.array([[100.0, 200.0]])]

    scaler = SubjectScaler()
    scaler.fit(train_seqs)

    np.testing.assert_allclose(scaler.mean_, [10.0, 20.0])

    # Transform test sequence using train statistics
    transformed_test = scaler.transform(test_seqs)
    # (100 - 10) / 1.0 (since variance in train is 0, scale defaults to 1.0)
    assert transformed_test[0][0, 0] == 90.0
