"""Anti-aliasing filtering and decimation for accelerometer signals."""

import numpy as np
from scipy import signal

from config import (
    BUTTERWORTH_CUTOFF_HZ,
    BUTTERWORTH_ORDER,
    DECIMATION_FACTOR,
    ORIGINAL_SAMPLING_RATE_HZ,
)


def design_anti_aliasing_filter(
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    fs: float = ORIGINAL_SAMPLING_RATE_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a digital Butterworth low-pass filter for anti-aliasing.

    Args:
        cutoff_hz: Filter -3dB cutoff frequency in Hz. Must be < fs / (2 * M).
        fs: Original sampling frequency in Hz (PAMAP2 is 100 Hz).
        order: Filter order (default: 4).

    Returns:
        (b, a) filter numerator and denominator polynomials.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    if not (0.0 < normal_cutoff < 1.0):
        raise ValueError(
            f"Normalized cutoff {normal_cutoff} must be in (0, 1). Cutoff: {cutoff_hz}, Nyquist: {nyquist}"
        )
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_anti_aliasing_filter(
    data: np.ndarray,
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    fs: float = ORIGINAL_SAMPLING_RATE_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter along axis 0.

    Uses forward-backward filtering (filtfilt) to ensure exact zero phase distortion,
    preserving peak alignment in time-series dynamics.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_samples = data.shape[0]
    min_samples = 3 * (max(order, 4) + 1)
    if n_samples < min_samples:
        # Too short to apply filtfilt without edge distortion; return copy
        return data.copy()

    b, a = design_anti_aliasing_filter(cutoff_hz, fs, order)
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered


def decimate_signal(
    data: np.ndarray,
    factor: int = DECIMATION_FACTOR,
    apply_filter: bool = True,
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    fs: float = ORIGINAL_SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Decimate signal by an integer factor with optional anti-aliasing.

    Args:
        data: Array of shape (N, D) or (N,).
        factor: Decimation factor M (e.g., 4 for 100 Hz -> 25 Hz).
        apply_filter: Whether to apply low-pass filter prior to sub-sampling.
        cutoff_hz: Anti-aliasing cutoff frequency.
        fs: Input sampling frequency.

    Returns:
        Decimated signal of shape (ceil(N / factor), D).
    """
    if factor <= 1:
        return data.copy()

    if apply_filter:
        filtered = apply_anti_aliasing_filter(data, cutoff_hz=cutoff_hz, fs=fs)
    else:
        filtered = data

    return filtered[::factor]
