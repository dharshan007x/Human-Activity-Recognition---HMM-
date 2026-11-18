"""Posterior state probability decoding (Smoothing) via Forward-Backward variables."""

import numpy as np


def compute_posteriors_log_space(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_likelihood: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior state probabilities gamma_t(i) = P(q_t = S_i | O, lambda) in log space.

    Equation:
        log gamma_t(i) = log alpha_t(i) + log beta_t(i) - log P(O | lambda)
        gamma_t(i) = exp(log gamma_t(i))

    CRITICAL THEORETICAL NOTE FOR VIVA DEFENSE:
        The sequence of pointwise argmax posteriors:
            q_hat_t = argmax_i gamma_t(i)
        is the INDIVIDUALLY most likely state at each individual time step.
        It does NOT maximize the joint probability P(q_1, ..., q_T | O, lambda).
        It can produce kinematically forbidden state transitions (where A_{i, j} = 0).
        The globally optimal trajectory of states is found ONLY by the Viterbi Algorithm.

    Args:
        log_alpha: Trellis matrix from forward_log_space, shape (T, N).
        log_beta: Trellis matrix from backward_log_space, shape (T, N).
        log_likelihood: Sequence log-likelihood log P(O | lambda).

    Returns:
        gamma: Posterior probability matrix of shape (T, N), where each row sums to 1.0.
        pointwise_states: Integer state indices of shape (T,) via pointwise argmax.
    """
    T, N = log_alpha.shape
    log_gamma = log_alpha + log_beta - log_likelihood

    # Exponentiate to get linear probabilities
    gamma = np.exp(log_gamma)

    # Normalize each row to strictly sum to 1.0 (correcting for minor floating point rounding)
    row_sums = np.sum(gamma, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    gamma = gamma / row_sums

    # Pointwise argmax (individually most likely state per time step)
    pointwise_states = np.argmax(gamma, axis=1)

    return gamma, pointwise_states


def compute_posteriors_scaled(
    alpha_hat: np.ndarray,
    beta_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior probabilities from scaled forward and backward variables.

    In Rabiner's scaled formulation:
        gamma_t(i) = (alpha_hat_t(i) * beta_hat_t(i)) / sum_j(alpha_hat_t(j) * beta_hat_t(j))

    Args:
        alpha_hat: Scaled forward trellis, shape (T, N).
        beta_hat: Scaled backward trellis, shape (T, N).

    Returns:
        gamma: Posterior probability matrix of shape (T, N).
        pointwise_states: Pointwise argmax state indices.
    """
    prod = alpha_hat * beta_hat
    row_sums = np.sum(prod, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    gamma = prod / row_sums

    pointwise_states = np.argmax(gamma, axis=1)
    return gamma, pointwise_states
