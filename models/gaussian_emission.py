"""Multivariate Gaussian emission distribution using Cholesky log-pdf in pure NumPy."""

import numpy as np
from scipy import linalg


class GaussianEmission:
    """Multivariate Gaussian emission density estimator for HMM states.

    Implements numerically stable Cholesky-based log-pdf evaluation
    and regularized maximum likelihood parameter estimation.
    """

    def __init__(
        self,
        n_features: int,
        covariance_regularization: float = 1e-4,
    ) -> None:
        self.n_features = n_features
        self.reg = covariance_regularization
        self.mean: np.ndarray | None = None  # Shape: (D,)
        self.cov: np.ndarray | None = None  # Shape: (D, D)
        self.chol_L: np.ndarray | None = None  # Lower triangular Cholesky: L @ L.T = Cov
        self.log_det_cov: float = 0.0

    def fit(self, observations: np.ndarray) -> "GaussianEmission":
        """Estimate mean vector and regularized covariance matrix from observations.

        Args:
            observations: Array of shape (N, D).
        """
        if observations.ndim != 2 or observations.shape[1] != self.n_features:
            raise ValueError(
                f"Expected observations of shape (N, {self.n_features}), got {observations.shape}"
            )
        if observations.shape[0] == 0:
            raise ValueError("Cannot fit GaussianEmission on empty observations.")

        self.mean = np.mean(observations, axis=0)
        diff = observations - self.mean
        n = observations.shape[0]

        # Empirical covariance with regularizing jitter eps * I
        emp_cov = (diff.T @ diff) / max(1, n)
        self.cov = emp_cov + self.reg * np.eye(self.n_features)

        # Cholesky factorization: Cov = L @ L.T
        try:
            self.chol_L = linalg.cholesky(self.cov, lower=True)
        except linalg.LinAlgError:
            # If still not positive-definite due to extreme degeneracy, boost regularization
            self.cov += 1e-2 * np.eye(self.n_features)
            self.chol_L = linalg.cholesky(self.cov, lower=True)

        # log|Cov| = 2 * sum(log(diag(L)))
        self.log_det_cov = float(2.0 * np.sum(np.log(np.diag(self.chol_L))))
        return self

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute multivariate Gaussian log probability density for sequence x.

        Vectorized over time steps T using forward substitution:
        L @ v = (x - mu).T  =>  (x - mu) @ Cov^-1 @ (x - mu).T = ||v||^2

        Args:
            x: Array of shape (T, D) or (D,).

        Returns:
            log_density: Array of shape (T,) or scalar float.
        """
        if self.mean is None or self.chol_L is None:
            raise RuntimeError("GaussianEmission must be fit before calling log_pdf.")

        is_1d = x.ndim == 1
        if is_1d:
            x = x[np.newaxis, :]

        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_features} features, got {x.shape[1]}"
            )

        # Difference vector: shape (T, D)
        diff = x - self.mean

        # Forward solve L @ v = diff.T (shape D x T)
        # scipy.linalg.solve_triangular is highly optimized BLAS/LAPACK call
        v = linalg.solve_triangular(self.chol_L, diff.T, lower=True, check_finite=False)

        # Mahalanobis distance squared: sum_d v[d, t]^2 for each t (shape T,)
        mahalanobis_sq = np.sum(v**2, axis=0)

        # log N(x | mu, Cov) = -0.5 * [ D * log(2*pi) + log|Cov| + Mahalanobis^2 ]
        log_density = -0.5 * (
            self.n_features * np.log(2.0 * np.pi) + self.log_det_cov + mahalanobis_sq
        )

        return log_density[0] if is_1d else log_density

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density in linear space."""
        return np.exp(self.log_pdf(x))
