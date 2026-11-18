"""Validation tests for Cholesky-based multivariate Gaussian emission against scipy."""

import numpy as np
import pytest
from scipy import stats

from models.gaussian_emission import GaussianEmission


def test_gaussian_emission_against_scipy() -> None:
    """Verify Cholesky log-pdf matches scipy.stats.multivariate_normal to machine precision."""
    rng = np.random.default_rng(42)
    n_features = 4
    n_samples = 200

    # Generate synthetic training cluster
    true_mean = np.array([0.5, 9.8, 1.2, 9.9])
    A = rng.standard_normal((n_features, n_features))
    true_cov = A @ A.T + 0.1 * np.eye(n_features)

    train_data = rng.multivariate_normal(true_mean, true_cov, size=n_samples)

    # Fit our pure NumPy Cholesky GaussianEmission
    model = GaussianEmission(n_features=n_features, covariance_regularization=1e-5)
    model.fit(train_data)

    # Evaluate on test query vectors
    test_data = rng.multivariate_normal(true_mean, true_cov, size=50)

    our_logpdf = model.log_pdf(test_data)
    scipy_logpdf = stats.multivariate_normal.logpdf(test_data, mean=model.mean, cov=model.cov)

    # Assert tolerance < 1e-10
    max_diff = np.max(np.abs(our_logpdf - scipy_logpdf))
    print(f"\n[TEST] Max diff between Cholesky log-pdf and scipy: {max_diff:.2e}")
    assert max_diff < 1e-10, f"Discrepancy {max_diff} exceeds 1e-10"


def test_gaussian_emission_dimension_mismatch() -> None:
    """Ensure wrong input dimension raises ValueError."""
    model = GaussianEmission(n_features=4)
    model.fit(np.zeros((10, 4)))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        model.log_pdf(np.zeros((5, 3)))
