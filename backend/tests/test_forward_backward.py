"""Core mathematical tests for Forward and Backward algorithms equality and duality."""

import numpy as np
import pytest

from models.backward import backward_log_space, backward_scaled
from models.forward import forward_log_space, forward_scaled
from models.posterior import compute_posteriors_log_space, compute_posteriors_scaled


@pytest.fixture
def sample_hmm_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixture providing valid 3-state HMM parameters and sequence emissions."""
    rng = np.random.default_rng(123)
    N = 3
    T = 60

    # Initial distribution pi
    raw_pi = rng.uniform(0.1, 1.0, size=N)
    pi = raw_pi / np.sum(raw_pi)

    # Transition matrix A (each row sums to 1)
    raw_A = rng.uniform(0.1, 1.0, size=(N, N))
    A = raw_A / np.sum(raw_A, axis=1, keepdims=True)

    # Random emission probabilities B (T x N)
    raw_B = rng.uniform(0.01, 1.0, size=(T, N))
    B = raw_B / np.sum(raw_B, axis=1, keepdims=True)

    return pi, A, B


def test_forward_equals_backward_log_space(sample_hmm_parameters) -> None:
    """MANDATORY VALIDATION: Forward log P(O|lambda) == Backward log P(O|lambda).

    Mathematical invariant:
    sum_i alpha_T(i) == sum_i pi_i * b_i(O_1) * beta_1(i)
    In log-space: abs(log_ll_forward - log_ll_backward) < 1e-8.
    """
    pi, A, B = sample_hmm_parameters
    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    ll_forward, log_alpha = forward_log_space(log_pi, log_A, log_B)
    ll_backward, log_beta = backward_log_space(log_pi, log_A, log_B)

    abs_diff = abs(ll_forward - ll_backward)
    print(f"\n[TEST] Forward log-likelihood:  {ll_forward:.10f}")
    print(f"[TEST] Backward log-likelihood: {ll_backward:.10f}")
    print(f"[TEST] Absolute discrepancy:    {abs_diff:.2e}")

    assert abs_diff < 1e-8, f"Forward and Backward log-likelihoods differ by {abs_diff:.2e} >= 1e-8"


def test_log_space_vs_scaled_agreement(sample_hmm_parameters) -> None:
    """Assert Log-Space and Scaled Normalization yield identical log-likelihoods."""
    pi, A, B = sample_hmm_parameters
    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    # Forward log vs scaled
    ll_fwd_log, _ = forward_log_space(log_pi, log_A, log_B)
    ll_fwd_scaled, alpha_hat, c_scale = forward_scaled(pi, A, B)

    # Backward scaled
    ll_bwd_scaled, beta_hat = backward_scaled(pi, A, B, c_scale)

    diff_fwd = abs(ll_fwd_log - ll_fwd_scaled)
    diff_bwd = abs(ll_fwd_scaled - ll_bwd_scaled)

    print(f"\n[TEST] Log-space Forward LL: {ll_fwd_log:.10f}")
    print(f"[TEST] Scaled Forward LL:    {ll_fwd_scaled:.10f}")
    print(f"[TEST] Scaled Backward LL:   {ll_bwd_scaled:.10f}")
    print(f"[TEST] Diff (log vs scaled): {diff_fwd:.2e}")

    assert diff_fwd < 1e-6, f"Log vs Scaled forward differs by {diff_fwd:.2e}"
    assert diff_bwd < 1e-8, f"Scaled forward vs Scaled backward differs by {diff_bwd:.2e}"


def test_posteriors_row_sum_to_one(sample_hmm_parameters) -> None:
    """Verify posterior state probabilities gamma_t(i) strictly sum to 1.0 for every t."""
    pi, A, B = sample_hmm_parameters
    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    ll, log_alpha = forward_log_space(log_pi, log_A, log_B)
    _, log_beta = backward_log_space(log_pi, log_A, log_B)

    gamma, states = compute_posteriors_log_space(log_alpha, log_beta, ll)

    # Check each row sums to 1.0 within 1e-12
    row_sums = np.sum(gamma, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # Check scaled posteriors
    ll_s, alpha_hat, c_scale = forward_scaled(pi, A, B)
    _, beta_hat = backward_scaled(pi, A, B, c_scale)
    gamma_scaled, states_scaled = compute_posteriors_scaled(alpha_hat, beta_hat)

    row_sums_scaled = np.sum(gamma_scaled, axis=1)
    np.testing.assert_allclose(row_sums_scaled, 1.0, atol=1e-12)

    # Both numerical methods should give virtually identical posteriors
    max_gamma_diff = np.max(np.abs(gamma - gamma_scaled))
    print(f"[TEST] Max diff between log-space gamma and scaled gamma: {max_gamma_diff:.2e}")
    assert max_gamma_diff < 1e-6
