"""Analytical hand-calculated 2-state 3-step benchmark test."""

import numpy as np

from models.backward import backward_log_space, backward_scaled
from models.forward import forward_log_space, forward_scaled
from models.posterior import compute_posteriors_log_space


def test_hand_calculated_2state_3step_hmm() -> None:
    """Verify Forward, Backward, and Posterior algorithms against exact hand-calculated derivation.

    Specification:
    States: N=2, Steps: T=3
    pi = [0.6, 0.4]
    A = [[0.7, 0.3],
         [0.4, 0.6]]
    B = [[0.5, 0.1],   # t=1
         [0.4, 0.7],   # t=2
         [0.8, 0.2]]   # t=3

    Exact Analytical Results:
    alpha_1 = [0.30, 0.04]
    alpha_2 = [0.0904, 0.0798]
    alpha_3 = [0.07616, 0.01500]
    P(O|lambda) = 0.07616 + 0.01500 = 0.09116
    log P(O|lambda) = ln(0.09116) = -2.395128795738072

    beta_3 = [1.0, 1.0]
    beta_2 = [0.62, 0.44]
    beta_1 = [0.2660, 0.2840]
    P_bwd(O|lambda) = 0.6*0.5*0.2660 + 0.4*0.1*0.2840 = 0.09116
    """
    pi = np.array([0.6, 0.4], dtype=np.float64)
    A = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    B = np.array([[0.5, 0.1], [0.4, 0.7], [0.8, 0.2]], dtype=np.float64)

    expected_P = 0.09116
    expected_log_P = np.log(expected_P)

    expected_alpha = np.array(
        [
            [0.30, 0.04],
            [0.0904, 0.0798],
            [0.07616, 0.01500],
        ]
    )

    expected_beta = np.array(
        [
            [0.2660, 0.2840],
            [0.62, 0.44],
            [1.0, 1.0],
        ]
    )

    expected_gamma = np.array(
        [
            [0.07980 / 0.09116, 0.01136 / 0.09116],
            [0.056048 / 0.09116, 0.035112 / 0.09116],
            [0.07616 / 0.09116, 0.01500 / 0.09116],
        ]
    )

    # 1. Test Log-Space Forward
    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    ll_fwd, log_alpha = forward_log_space(log_pi, log_A, log_B)
    np.testing.assert_allclose(ll_fwd, expected_log_P, rtol=1e-12)
    np.testing.assert_allclose(np.exp(log_alpha), expected_alpha, rtol=1e-12)

    # 2. Test Log-Space Backward
    ll_bwd, log_beta = backward_log_space(log_pi, log_A, log_B)
    np.testing.assert_allclose(ll_bwd, expected_log_P, rtol=1e-12)
    np.testing.assert_allclose(np.exp(log_beta), expected_beta, rtol=1e-12)

    # 3. Test Scaled Forward & Backward
    ll_scaled_fwd, alpha_hat, c_scale = forward_scaled(pi, A, B)
    np.testing.assert_allclose(ll_scaled_fwd, expected_log_P, rtol=1e-12)

    ll_scaled_bwd, beta_hat = backward_scaled(pi, A, B, c_scale)
    np.testing.assert_allclose(ll_scaled_bwd, expected_log_P, rtol=1e-12)

    # 4. Test Posterior Decoding gamma
    gamma, states = compute_posteriors_log_space(log_alpha, log_beta, ll_fwd)
    np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-12)

    # Pointwise state sequence: state 0 at t=0, state 0 at t=1, state 0 at t=2
    expected_states = np.array([0, 0, 0])
    np.testing.assert_array_equal(states, expected_states)

    print("\n[SUCCESS] 2-state 3-step hand-calculated benchmark passed with 1e-12 precision!")
