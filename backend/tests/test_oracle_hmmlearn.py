"""Independent oracle cross-check of our pure NumPy HMM against hmmlearn.

MANDATORY PROJECT RULE:
hmmlearn is utilized EXCLUSIVELY inside this test module as an external oracle
benchmark to rigorously cross-verify our custom NumPy implementation.
It is NOT used anywhere in the core models or inference application.
"""

import numpy as np
from hmmlearn import hmm

from models.hmm_model import GaussianHMM


def test_custom_hmm_against_hmmlearn_oracle() -> None:
    """Verify our pure NumPy Forward, Backward, Posterior and Viterbi match hmmlearn oracle."""
    rng = np.random.default_rng(999)
    n_states = 3
    n_features = 4
    T = 40

    # 1. Define common parameters
    raw_pi = rng.uniform(0.2, 1.0, size=n_states)
    pi = raw_pi / np.sum(raw_pi)

    raw_A = rng.uniform(0.1, 1.0, size=(n_states, n_states))
    A = raw_A / np.sum(raw_A, axis=1, keepdims=True)

    means = rng.uniform(-2.0, 2.0, size=(n_states, n_features))
    covars = np.zeros((n_states, n_features, n_features))
    for s in range(n_states):
        mat = rng.standard_normal((n_features, n_features))
        covars[s] = mat @ mat.T + 0.5 * np.eye(n_features)

    # 2. Setup hmmlearn oracle model
    oracle = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        init_params="",
    )
    oracle.startprob_ = pi.copy()
    oracle.transmat_ = A.copy()
    oracle.means_ = means.copy()
    oracle.covars_ = covars.copy()

    # 3. Setup our custom pure NumPy model
    custom = GaussianHMM(n_states=n_states, n_features=n_features, state_names=["W", "S", "R"])
    custom.pi = pi.copy()
    custom.A = A.copy()
    for s in range(n_states):
        custom.emissions[s].mean = means[s].copy()
        custom.emissions[s].cov = covars[s].copy()
        custom.emissions[s].chol_L = np.linalg.cholesky(covars[s])
        custom.emissions[s].log_det_cov = float(
            2.0 * np.sum(np.log(np.diag(custom.emissions[s].chol_L)))
        )
    custom.is_fitted = True

    # 4. Generate random test sequence
    test_seq = rng.standard_normal((T, n_features))

    # --- Benchmark 1: Sequence Log-Likelihood (Forward Algorithm) ---
    custom_ll = custom.score(test_seq)
    oracle_ll = float(oracle.score(test_seq))
    diff_ll = abs(custom_ll - oracle_ll)
    print(f"\n[ORACLE TEST] Custom Log-Likelihood: {custom_ll:.8f}")
    print(f"[ORACLE TEST] Oracle Log-Likelihood: {oracle_ll:.8f}")
    print(f"[ORACLE TEST] Absolute Discrepancy: {diff_ll:.2e}")
    assert diff_ll < 1e-8, f"Log-likelihood diff {diff_ll} exceeds 1e-8"

    # --- Benchmark 2: Posterior State Probabilities (Gamma) ---
    custom_gamma, _, _, _, _ = custom.posterior(test_seq, method="log")
    oracle_gamma = oracle.predict_proba(test_seq)
    max_gamma_diff = np.max(np.abs(custom_gamma - oracle_gamma))
    print(f"[ORACLE TEST] Max gamma posterior diff: {max_gamma_diff:.2e}")
    assert max_gamma_diff < 1e-8, f"Posterior gamma diff {max_gamma_diff} exceeds 1e-8"

    # --- Benchmark 3: Viterbi Decoding (Globally optimal path) ---
    custom_viterbi_path, _, _ = custom.decode_viterbi(test_seq)
    oracle_viterbi_path = oracle.predict(test_seq)
    mismatches = np.sum(custom_viterbi_path != oracle_viterbi_path)
    print(f"[ORACLE TEST] Viterbi path mismatches: {mismatches} / {T}")
    assert mismatches == 0, "Viterbi path from custom model does not match hmmlearn oracle!"
