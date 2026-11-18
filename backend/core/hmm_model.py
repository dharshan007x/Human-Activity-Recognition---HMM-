"""Full Gaussian Hidden Markov Model class in pure NumPy."""

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from backend.config import (
        COVARIANCE_REGULARIZATION,
        FEATURE_NAMES,
        LAPLACE_SMOOTHING,
        NUM_STATES,
        SAVED_HMM_PATH,
        STATE_NAMES,
    )
    from backend.core.backward import backward_log_space, backward_scaled
    from backend.core.forward import forward_log_space, forward_scaled
    from backend.core.gaussian_emission import GaussianEmission
    from backend.core.posterior import compute_posteriors_log_space, compute_posteriors_scaled
    from backend.core.viterbi import viterbi_log_space
except ImportError:
    from config import (
        COVARIANCE_REGULARIZATION,
        FEATURE_NAMES,
        LAPLACE_SMOOTHING,
        NUM_STATES,
        SAVED_HMM_PATH,
        STATE_NAMES,
    )
    from models.backward import backward_log_space, backward_scaled
    from models.forward import forward_log_space, forward_scaled
    from models.gaussian_emission import GaussianEmission
    from models.posterior import compute_posteriors_log_space, compute_posteriors_scaled
    from models.viterbi import viterbi_log_space


class EmptySequenceError(ValueError):
    """Raised when an empty observation sequence is provided."""

    pass


class DimensionMismatchError(ValueError):
    """Raised when observation dimensions do not match model features."""

    pass


class InvalidStateError(ValueError):
    """Raised when an invalid state index or name is referenced."""

    pass


class GaussianHMM:
    """Supervised Gaussian Hidden Markov Model for Human Activity Recognition.

    States: Walking, Sitting, Running.
    Observations: Continuous triaxial acceleration + Euclidean magnitude.

    Core inference algorithms implemented from first principles in pure NumPy:
    - Forward algorithm (log-space and scaled)
    - Backward algorithm (log-space and scaled)
    - Posterior state probabilities (Smoothing)
    - Dynamic programming Viterbi decoding (Extension)
    """

    def __init__(
        self,
        n_states: int = NUM_STATES,
        n_features: int = len(FEATURE_NAMES),
        state_names: list[str] | None = None,
        reg_cov: float = COVARIANCE_REGULARIZATION,
        laplace_smoothing: float = LAPLACE_SMOOTHING,
    ) -> None:
        self.n_states = n_states
        self.n_features = n_features
        self.state_names = state_names or list(STATE_NAMES)
        self.reg_cov = reg_cov
        self.laplace_smoothing = laplace_smoothing

        # Model parameters
        self.pi: np.ndarray = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        self.A: np.ndarray = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
        self.emissions: list[GaussianEmission] = [
            GaussianEmission(n_features=self.n_features, covariance_regularization=self.reg_cov)
            for _ in range(n_states)
        ]
        self.is_fitted: bool = False
        self.metadata: dict[str, Any] = {}

    @property
    def means(self) -> np.ndarray:
        return np.array([em.mean for em in self.emissions], dtype=np.float64)

    @property
    def covars(self) -> np.ndarray:
        return np.array([em.cov for em in self.emissions], dtype=np.float64)

    def fit(
        self,
        sequences: list[np.ndarray],
        state_sequences: list[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> "GaussianHMM":
        """Supervised Maximum Likelihood parameter estimation from labeled sequences.

        CRITICAL THEORETICAL PRINCIPLE FOR VIVA DEFENSE:
            The Forward and Backward algorithms are INFERENCE algorithms used to calculate
            sequence likelihoods and state posteriors on already-parameterized HMMs.
            They DO NOT estimate or learn parameters.
            Parameters here are estimated via supervised Maximum Likelihood Estimation (MLE)
            with Laplace smoothing on discrete counts and regularized Gaussian estimation.

        Args:
            sequences: List of arrays of shape (T_m, D).
            state_sequences: List of integer state label arrays of shape (T_m,).
            metadata: Optional dictionary of provenance records.
        """
        if len(sequences) == 0 or len(state_sequences) == 0:
            raise EmptySequenceError("Cannot fit GaussianHMM on empty sequence lists.")
        if len(sequences) != len(state_sequences):
            raise ValueError(
                "Number of observation sequences must match number of label sequences."
            )

        # 1. Initial State Distribution pi with Laplace smoothing
        init_counts = np.zeros(self.n_states, dtype=np.float64)
        for states in state_sequences:
            if len(states) > 0:
                init_counts[states[0]] += 1.0

        self.pi = (init_counts + self.laplace_smoothing) / (
            np.sum(init_counts) + self.n_states * self.laplace_smoothing
        )

        # 2. Transition Probability Matrix A with Laplace smoothing
        trans_counts = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for states in state_sequences:
            for t in range(len(states) - 1):
                i_from = states[t]
                j_to = states[t + 1]
                trans_counts[i_from, j_to] += 1.0

        for i in range(self.n_states):
            row_sum = np.sum(trans_counts[i, :])
            self.A[i, :] = (trans_counts[i, :] + self.laplace_smoothing) / (
                row_sum + self.n_states * self.laplace_smoothing
            )

        # 3. Multivariate Gaussian Emission Distributions
        # Pool all observations grouped by state label
        state_observations: list[list[np.ndarray]] = [[] for _ in range(self.n_states)]
        for seq, states in zip(sequences, state_sequences):
            for t in range(len(states)):
                s = states[t]
                state_observations[s].append(seq[t])

        for state_idx in range(self.n_states):
            if len(state_observations[state_idx]) == 0:
                raise ValueError(
                    f"No observations available to estimate emission for state '{self.state_names[state_idx]}'"
                )
            obs_matrix = np.array(state_observations[state_idx], dtype=np.float64)
            self.emissions[state_idx].fit(obs_matrix)

        self.is_fitted = True
        self.metadata = metadata or {}
        return self

    def _validate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Validate input sequence dimensions and non-emptiness."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit or loaded before running inference.")

        seq = np.asarray(sequence, dtype=np.float64)
        if seq.ndim != 2:
            raise DimensionMismatchError(
                f"Expected 2D array of shape (T, {self.n_features}), got {seq.shape}"
            )
        if seq.shape[0] == 0:
            raise EmptySequenceError("Observation sequence cannot be empty (T=0).")
        if seq.shape[1] != self.n_features:
            raise DimensionMismatchError(
                f"Feature dimension mismatch: expected {self.n_features}, got {seq.shape[1]}"
            )
        if not np.all(np.isfinite(seq)):
            raise ValueError("Observation sequence contains NaN or Infinite values.")

        return seq

    def compute_log_emissions(self, sequence: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities log b_j(O_t) for all states j and time steps t.

        Returns:
            log_B: Matrix of shape (T, N).
        """
        seq = self._validate_sequence(sequence)
        T = seq.shape[0]
        log_B = np.zeros((T, self.n_states), dtype=np.float64)

        for j in range(self.n_states):
            log_B[:, j] = self.emissions[j].log_pdf(seq)

        return log_B

    def forward(
        self, sequence: np.ndarray, method: str = "log"
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Run the Forward Algorithm.

        Args:
            sequence: Array of shape (T, D).
            method: 'log' for pure log-space or 'scaled' for Rabiner scaling.

        Returns:
            (log_likelihood, alpha_trellis, scale_factors_or_none)
        """
        seq = self._validate_sequence(sequence)
        log_B = self.compute_log_emissions(seq)

        if method == "log":
            log_pi = np.log(self.pi)
            log_A = np.log(self.A)
            ll, log_alpha = forward_log_space(log_pi, log_A, log_B)
            return ll, log_alpha, None
        elif method == "scaled":
            B = np.exp(log_B)
            ll, alpha_hat, c_scale = forward_scaled(self.pi, self.A, B)
            return ll, alpha_hat, c_scale
        else:
            raise ValueError(f"Unknown forward method: '{method}'. Choose 'log' or 'scaled'.")

    def backward(
        self,
        sequence: np.ndarray,
        method: str = "log",
        c_scale: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Run the Backward Algorithm.

        Args:
            sequence: Array of shape (T, D).
            method: 'log' or 'scaled'.
            c_scale: Required if method='scaled' (from forward_scaled).

        Returns:
            (log_likelihood, beta_trellis)
        """
        seq = self._validate_sequence(sequence)
        log_B = self.compute_log_emissions(seq)

        if method == "log":
            log_pi = np.log(self.pi)
            log_A = np.log(self.A)
            ll, log_beta = backward_log_space(log_pi, log_A, log_B)
            return ll, log_beta
        elif method == "scaled":
            if c_scale is None:
                _, _, c_scale = forward_scaled(self.pi, self.A, np.exp(log_B))
            B = np.exp(log_B)
            ll, beta_hat = backward_scaled(self.pi, self.A, B, c_scale)
            return ll, beta_hat
        else:
            raise ValueError(f"Unknown backward method: '{method}'. Choose 'log' or 'scaled'.")

    def posterior(
        self, sequence: np.ndarray, method: str = "log"
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """Compute state posteriors gamma_t(i) = P(q_t = S_i | O, lambda) and pointwise argmax.

        Returns:
            (gamma, pointwise_states, log_likelihood, alpha_trellis, beta_trellis)
        """
        seq = self._validate_sequence(sequence)

        if method == "log":
            ll_fwd, log_alpha, _ = self.forward(seq, method="log")
            ll_bwd, log_beta = self.backward(seq, method="log")
            gamma, states = compute_posteriors_log_space(log_alpha, log_beta, ll_fwd)
            return gamma, states, ll_fwd, log_alpha, log_beta
        elif method == "scaled":
            ll_fwd, alpha_hat, c_scale = self.forward(seq, method="scaled")
            ll_bwd, beta_hat = self.backward(seq, method="scaled", c_scale=c_scale)
            gamma, states = compute_posteriors_scaled(alpha_hat, beta_hat)
            return gamma, states, ll_fwd, alpha_hat, beta_hat
        else:
            raise ValueError(f"Unknown posterior method: '{method}'.")

    def score(self, sequence: np.ndarray) -> float:
        """Compute log P(O | lambda) via the Forward algorithm."""
        ll, _, _ = self.forward(sequence, method="log")
        return ll

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Predict activity states for each time step via pointwise argmax of posterior gamma."""
        _, states, _, _, _ = self.posterior(sequence, method="log")
        return states

    def decode_viterbi(self, sequence: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Decode the single most likely path of hidden states via Viterbi (Extension)."""
        seq = self._validate_sequence(sequence)
        log_B = self.compute_log_emissions(seq)
        log_pi = np.log(self.pi)
        log_A = np.log(self.A)
        return viterbi_log_space(log_pi, log_A, log_B)

    def save(self, file_path: Path | str = SAVED_HMM_PATH) -> None:
        """Serialize complete model parameters to JSON with provenance metadata."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        emissions_data = []
        for j in range(self.n_states):
            m = self.emissions[j].mean
            c = self.emissions[j].cov
            emissions_data.append(
                {
                    "state_name": self.state_names[j],
                    "mean": m.tolist() if m is not None else [],
                    "cov": c.tolist() if c is not None else [],
                    "log_det_cov": self.emissions[j].log_det_cov,
                }
            )

        payload = {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "state_names": self.state_names,
            "reg_cov": self.reg_cov,
            "laplace_smoothing": self.laplace_smoothing,
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "emissions": emissions_data,
            "metadata": self.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: Path | str = SAVED_HMM_PATH) -> "GaussianHMM":
        """Deserialize GaussianHMM from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        model = cls(
            n_states=payload["n_states"],
            n_features=payload["n_features"],
            state_names=payload["state_names"],
            reg_cov=payload.get("reg_cov", 1e-4),
            laplace_smoothing=payload.get("laplace_smoothing", 1.0),
        )

        model.pi = np.array(payload["pi"], dtype=np.float64)
        model.A = np.array(payload["A"], dtype=np.float64)

        for j, em_data in enumerate(payload["emissions"]):
            em = model.emissions[j]
            em.mean = np.array(em_data["mean"], dtype=np.float64)
            em.cov = np.array(em_data["cov"], dtype=np.float64)
            em.log_det_cov = float(em_data["log_det_cov"])
            em.chol_L = np.linalg.cholesky(em.cov)

        model.is_fitted = True
        model.metadata = payload.get("metadata", {})
        return model
