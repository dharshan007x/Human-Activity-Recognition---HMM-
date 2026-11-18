"""Property-based testing using Hypothesis for invariant validation."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from models.backward import backward_log_space
from models.forward import forward_log_space, numpy_logsumexp
from models.posterior import compute_posteriors_log_space


@settings(max_examples=50, deadline=None)
@given(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    ),
)
def test_logsumexp_shift_invariance(c: float, values: list[float]) -> None:
    """Property: logsumexp(x + c) == logsumexp(x) + c."""
    arr = np.array(values, dtype=np.float64)
    res_original = numpy_logsumexp(arr)
    res_shifted = numpy_logsumexp(arr + c)
    np.testing.assert_allclose(res_shifted, res_original + c, rtol=1e-7, atol=1e-7)


@settings(max_examples=30, deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=5, max_value=30),
)
def test_forward_backward_equality_random_hmm(n_states: int, t_steps: int) -> None:
    """Property: For any valid HMM and non-degenerate emissions, Forward LL == Backward LL."""
    rng = np.random.default_rng()

    # Random valid pi (sums to 1)
    raw_pi = rng.uniform(0.1, 1.0, size=n_states)
    pi = raw_pi / np.sum(raw_pi)

    # Random valid A (each row sums to 1)
    raw_A = rng.uniform(0.1, 1.0, size=(n_states, n_states))
    A = raw_A / np.sum(raw_A, axis=1, keepdims=True)

    # Random emission probabilities (bounded above 0 to prevent -inf)
    raw_B = rng.uniform(0.01, 1.0, size=(t_steps, n_states))
    B = raw_B / np.sum(raw_B, axis=1, keepdims=True)

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    ll_fwd, log_alpha = forward_log_space(log_pi, log_A, log_B)
    ll_bwd, log_beta = backward_log_space(log_pi, log_A, log_B)

    # Invariant: Forward log-likelihood == Backward log-likelihood
    assert abs(ll_fwd - ll_bwd) < 1e-7

    # Invariant: Posterior gamma rows sum to 1.0
    gamma, _ = compute_posteriors_log_space(log_alpha, log_beta, ll_fwd)
    row_sums = np.sum(gamma, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
