"""Backward Algorithm implementations: Log-Space and Scaled Normalization."""

import numpy as np

from models.forward import numpy_logsumexp


def backward_log_space(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Backward algorithm computed entirely in log space.

    Args:
        log_pi: Log initial distribution, shape (N,).
        log_A: Log transition probability matrix, shape (N, N), where log_A[i, j] = log P(S_j | S_i).
        log_B: Log emission probabilities, shape (T, N), where log_B[t, j] = log P(O_t | S_j).

    Returns:
        log_likelihood: Sequence log probability log P(O | lambda) evaluated at t = 1.
        log_beta: Complete backward trellis matrix of shape (T, N), where log_beta[t, i] = log beta_t(i).
    """
    T, N = log_B.shape
    if T == 0:
        raise ValueError("Sequence length T must be >= 1.")

    log_beta = np.zeros((T, N), dtype=np.float64)

    # 1. Initialization: beta_T(i) = 1 => log beta_T(i) = 0
    log_beta[T - 1, :] = 0.0

    # 2. Induction (backwards in time from T-2 down to 0)
    for t in range(T - 2, -1, -1):
        # We need log_beta[t, i] = logsumexp_j (log A_{ij} + log b_j(O_{t+1}) + log beta_{t+1}(j))
        # log_A has shape (N, N) where rows are i, cols are j.
        # log_B[t+1, :] + log_beta[t+1, :] has shape (N,) along j.
        terms = log_A + (log_B[t + 1, :] + log_beta[t + 1, :])[np.newaxis, :]
        log_beta[t, :] = numpy_logsumexp(terms, axis=1)

    # 3. Termination: log P(O | lambda) = logsumexp_i (log pi_i + log b_i(O_1) + log beta_1(i))
    init_terms = log_pi + log_B[0, :] + log_beta[0, :]
    log_likelihood = float(numpy_logsumexp(init_terms))

    return log_likelihood, log_beta


def backward_scaled(
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    c_scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Backward algorithm using Rabiner's per-step scaling factors from the forward pass.

    Ensures numerical values stay bounded and match scaled alpha for posterior decoding.

    Args:
        pi: Initial state distribution, shape (N,).
        A: Transition probability matrix, shape (N, N).
        B: Emission probabilities, shape (T, N).
        c_scale: Scale factors from forward_scaled, shape (T,).

    Returns:
        log_likelihood: Log likelihood computed via forward scaling factors.
        beta_hat: Scaled backward trellis matrix of shape (T, N).
    """
    T, N = B.shape
    if T == 0:
        raise ValueError("Sequence length T must be >= 1.")

    beta_hat = np.zeros((T, N), dtype=np.float64)

    # 1. Initialization: beta_T(i) = 1, scaled by c_T
    beta_hat[T - 1, :] = c_scale[T - 1]

    # 2. Induction backwards from T-2 down to 0
    for t in range(T - 2, -1, -1):
        # A @ (B[t+1, :] * beta_hat[t+1, :]) gives (N,)
        term = B[t + 1, :] * beta_hat[t + 1, :]
        beta_raw = A @ term
        beta_hat[t, :] = c_scale[t] * beta_raw

    # 3. Likelihood matches -sum(log(c_t))
    log_likelihood = float(-np.sum(np.log(c_scale)))
    return log_likelihood, beta_hat
