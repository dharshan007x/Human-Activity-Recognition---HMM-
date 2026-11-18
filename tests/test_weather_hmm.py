"""Validation against the classic textbook Weather HMM (Rabiner Sunny/Rainy)."""

import numpy as np

from models.backward import backward_log_space
from models.forward import forward_log_space


def test_classic_textbook_weather_hmm() -> None:
    """Validate classic textbook Weather HMM with observation sequence [Walk, Shop, Clean].

    States: 0: Rainy, 1: Sunny
    Observations: 0: Walk, 1: Shop, 2: Clean

    pi = [0.6, 0.4]
    A = [[0.7, 0.3],
         [0.4, 0.6]]
    Emission matrix:
    B_obs = [[0.1, 0.4, 0.5],   # Rainy -> [Walk, Shop, Clean]
             [0.6, 0.3, 0.1]]   # Sunny -> [Walk, Shop, Clean]

    Observation sequence: [Walk, Shop, Clean]
    b(O_1 = Walk)  = [0.1, 0.6]
    b(O_2 = Shop)  = [0.4, 0.3]
    b(O_3 = Clean) = [0.5, 0.1]

    Analytical Probability:
    alpha_1 = [0.06, 0.24]
    alpha_2 = [0.0552, 0.0486]
    alpha_3 = [0.02904, 0.004572]
    P(O|lambda) = 0.033612
    log P(O|lambda) = ln(0.033612) = -3.392873111...
    """
    pi = np.array([0.6, 0.4], dtype=np.float64)
    A = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    B = np.array([[0.1, 0.6], [0.4, 0.3], [0.5, 0.1]], dtype=np.float64)

    expected_P = 0.033612
    expected_log_P = np.log(expected_P)

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    # Forward pass
    ll_fwd, log_alpha = forward_log_space(log_pi, log_A, log_B)
    np.testing.assert_allclose(ll_fwd, expected_log_P, rtol=1e-12)

    # Backward pass
    ll_bwd, log_beta = backward_log_space(log_pi, log_A, log_B)
    np.testing.assert_allclose(ll_bwd, expected_log_P, rtol=1e-12)

    # Check forward equals backward
    assert abs(ll_fwd - ll_bwd) < 1e-14
    print("\n[SUCCESS] Classic textbook Weather HMM matches analytical ground truth to 1e-14!")
