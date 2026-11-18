"""Tests for input validation, schema enforcement, and typed exceptions."""

import numpy as np
import pytest

from models.hmm_model import (
    DimensionMismatchError,
    EmptySequenceError,
    GaussianHMM,
)


@pytest.fixture
def trained_dummy_model() -> GaussianHMM:
    """Provide a minimal trained 3-state HMM."""
    model = GaussianHMM(n_states=3, n_features=4)
    # Fit on minimal valid dummy data
    seqs = [np.zeros((5, 4))] * 3
    states = [np.array([0, 0, 1, 1, 2])] * 3
    model.fit(seqs, states)
    return model


def test_reject_empty_sequence(trained_dummy_model: GaussianHMM) -> None:
    """Empty observation array must raise EmptySequenceError."""
    with pytest.raises(EmptySequenceError, match="cannot be empty"):
        trained_dummy_model.forward(np.empty((0, 4)))


def test_reject_dimension_mismatch(trained_dummy_model: GaussianHMM) -> None:
    """Incorrect number of feature columns must raise DimensionMismatchError."""
    # Model expects 4 features; pass 3
    with pytest.raises(DimensionMismatchError, match="Feature dimension mismatch"):
        trained_dummy_model.forward(np.ones((10, 3)))


def test_reject_1d_input(trained_dummy_model: GaussianHMM) -> None:
    """1D array passed to forward must raise DimensionMismatchError."""
    with pytest.raises(DimensionMismatchError, match="Expected 2D array"):
        trained_dummy_model.forward(np.ones(4))


def test_reject_nan_inf_input(trained_dummy_model: GaussianHMM) -> None:
    """Sequences containing NaN or Inf must raise ValueError."""
    seq_nan = np.ones((5, 4))
    seq_nan[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        trained_dummy_model.forward(seq_nan)

    seq_inf = np.ones((5, 4))
    seq_inf[1, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Infinite"):
        trained_dummy_model.forward(seq_inf)
