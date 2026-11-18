"""Forward Algorithm implementations: Log-Space and Scaled Normalization."""

import numpy as np


def numpy_logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp in pure NumPy using the max-subtraction trick.

    log(sum(exp(a_i))) = m + log(sum(exp(a_i - m))), where m = max(a_i).
    """
    a = np.asarray(a)
    max_val = np.max(a, axis=axis, keepdims=True)
    if not np.all(np.isfinite(max_val)):
        with np.errstate(divide="ignore"):
            res = np.sum(np.exp(a), axis=axis, keepdims=keepdims)
            return np.log(res)

    diff = a - max_val
    sum_exp = np.sum(np.exp(diff), axis=axis, keepdims=True)
    lse = max_val + np.log(sum_exp)
    if not keepdims:
        lse = np.squeeze(lse, axis=axis) if axis is not None else np.squeeze(lse)
    return lse


def forward_log_space(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Forward algorithm computed entirely in log space.

    Guarantees no numerical underflow for arbitrarily long observation sequences.

    Args:
        log_pi: Log initial distribution, shape (N,).
        log_A: Log transition probability matrix, shape (N, N), where log_A[i, j] = log P(S_j | S_i).
        log_B: Log emission probabilities, shape (T, N), where log_B[t, j] = log P(O_t | S_j).

    Returns:
        log_likelihood: Sequence log probability log P(O | lambda).
        log_alpha: Complete forward trellis matrix of shape (T, N), where log_alpha[t, i] = log alpha_t(i).
    """
    T, N = log_B.shape
    if T == 0:
        raise ValueError("Sequence length T must be >= 1.")

    log_alpha = np.zeros((T, N), dtype=np.float64)

    # 1. Initialization: log alpha_1(i) = log pi_i + log b_i(O_1)
    log_alpha[0, :] = log_pi + log_B[0, :]

    # 2. Induction (explicit time loop for viva clarity and educational step-tracing)
    # Vectorized across states within each time step
    for t in range(0, T - 1):
        # Shape of log_alpha[t, :]: (N, 1) to broadcast across destination states j (columns)
        # log_alpha[t, :, None] + log_A has shape (N, N), where [i, j] = log alpha_t(i) + log A_{ij}
        accum = log_alpha[t, :, np.newaxis] + log_A
        # logsumexp over source states i (axis 0) gives shape (N,)
        log_alpha[t + 1, :] = numpy_logsumexp(accum, axis=0) + log_B[t + 1, :]

    # 3. Termination: log P(O | lambda) = logsumexp_i(log alpha_T(i))
    log_likelihood = float(numpy_logsumexp(log_alpha[T - 1, :]))
    return log_likelihood, log_alpha


def forward_scaled(
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Forward algorithm using Rabiner's per-step scaling factors.

    At each step t, alpha_t is normalized by c_t = 1 / sum_i(alpha_t(i)),
    ensuring all intermediate values sum to 1 and stay safely within dynamic range.

    Args:
        pi: Initial state distribution, shape (N,).
        A: Transition probability matrix, shape (N, N).
        B: Emission probabilities, shape (T, N).

    Returns:
        log_likelihood: Sequence log probability log P(O | lambda) = -sum(log(c_t)).
        alpha_hat: Scaled forward trellis matrix of shape (T, N).
        c_scale: Normalization scale factors array of shape (T,).
    """
    T, N = B.shape
    if T == 0:
        raise ValueError("Sequence length T must be >= 1.")

    alpha_hat = np.zeros((T, N), dtype=np.float64)
    c_scale = np.zeros(T, dtype=np.float64)

    # 1. Initialization
    alpha_1_raw = pi * B[0, :]
    sum_1 = np.sum(alpha_1_raw)
    if sum_1 <= 0.0:
        c_scale[0] = 1.0
        alpha_hat[0, :] = 1.0 / N
    else:
        c_scale[0] = 1.0 / sum_1
        alpha_hat[0, :] = c_scale[0] * alpha_1_raw

    # 2. Induction
    for t in range(0, T - 1):
        # alpha_hat[t, :] @ A gives 1D array of shape (N,)
        alpha_pred = alpha_hat[t, :] @ A
        alpha_t1_raw = alpha_pred * B[t + 1, :]
        sum_t1 = np.sum(alpha_t1_raw)

        if sum_t1 <= 0.0:
            c_scale[t + 1] = 1.0
            alpha_hat[t + 1, :] = 1.0 / N
        else:
            c_scale[t + 1] = 1.0 / sum_t1
            alpha_hat[t + 1, :] = c_scale[t + 1] * alpha_t1_raw

    # 3. Sequence Log-Likelihood: log P(O | lambda) = - sum_{t=1}^T log(c_t)
    log_likelihood = float(-np.sum(np.log(c_scale)))
    return log_likelihood, alpha_hat, c_scale
