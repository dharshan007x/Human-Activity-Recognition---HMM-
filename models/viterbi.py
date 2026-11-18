"""Viterbi decoding algorithm: globally optimal hidden state sequence (Extension)."""

import numpy as np


def viterbi_log_space(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find the single globally most likely hidden state sequence via dynamic programming.

    Contrasted with pointwise argmax of gamma posteriors:
    Viterbi guarantees that the decoded sequence maximizes the JOINT path probability
    P(q_1, ..., q_T, O | lambda) and strictly obeys transition constraints (A_{ij} > 0).

    Args:
        log_pi: Log initial distribution, shape (N,).
        log_A: Log transition probability matrix, shape (N, N).
        log_B: Log emission probabilities, shape (T, N).

    Returns:
        best_path: Globally optimal state sequence array of shape (T,).
        best_log_prob: Log probability of the best path.
        viterbi_trellis: Dynamic programming matrix V of shape (T, N).
    """
    T, N = log_B.shape
    if T == 0:
        raise ValueError("Sequence length T must be >= 1.")

    V = np.zeros((T, N), dtype=np.float64)
    backpointers = np.zeros((T, N), dtype=np.int32)

    # 1. Initialization
    V[0, :] = log_pi + log_B[0, :]

    # 2. Recursion
    for t in range(1, T):
        for j in range(N):
            # Candidate transitions from all states i at t-1 to state j at t
            candidates = V[t - 1, :] + log_A[:, j]
            best_prev = int(np.argmax(candidates))
            backpointers[t, j] = best_prev
            V[t, j] = candidates[best_prev] + log_B[t, j]

    # 3. Termination
    best_last_state = int(np.argmax(V[T - 1, :]))
    best_log_prob = float(V[T - 1, best_last_state])

    # 4. Backtracking path
    best_path = np.zeros(T, dtype=np.int32)
    best_path[T - 1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointers[t + 1, best_path[t + 1]]

    return best_path, best_log_prob, V
