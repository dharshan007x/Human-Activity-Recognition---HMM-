"""Page 12: About - Viva Voce Defense Guide, Mathematical Proofs, and Bibliography."""

import streamlit as st

st.title("Project Architecture & Viva Voce Defense")
st.caption("Theoretical proofs, academic bibliography, and oral defense answers")

st.markdown(
    """
<div class="paper-card">
    <h3 style="margin-top:0;">University Defense Questions & Answers (Top 6)</h3>
    <div style="margin-bottom: 1.25rem;">
        <h4>1. Why does Forward sequence likelihood equal Backward sequence likelihood?</h4>
        <p>
            <strong>Proof:</strong> The Forward variable computes $\\alpha_t(i) = P(O_1, \\dots, O_t, q_t = S_i \\mid \\lambda)$,
            so at the final step $T$, $\\sum_{i=1}^N \\alpha_T(i) = \\sum_{i=1}^N P(O, q_T = S_i \\mid \\lambda) = P(O \\mid \\lambda)$.<br>
            Conversely, the Backward variable computes $\\beta_t(i) = P(O_{t+1}, \\dots, O_T \\mid q_t = S_i, \\lambda)$.
            At step $t = 1$:
            $$\\sum_{i=1}^N \\pi_i b_i(O_1) \\beta_1(i) = \\sum_{i=1}^N P(q_1 = S_i) P(O_1 \\mid q_1 = S_i) P(O_2, \\dots, O_T \\mid q_1 = S_i) = \\sum_{i=1}^N P(O, q_1 = S_i \\mid \\lambda) = P(O \\mid \\lambda)$$
            Both marginalize over hidden state paths at opposite boundaries, yielding the exact identical sequence likelihood.
        </p>
    </div>
    <div style="margin-bottom: 1.25rem;">
        <h4>2. Why is the pointwise argmax of posterior γ_t NOT equivalent to the Viterbi path?</h4>
        <p>
            <strong>Answer:</strong> Pointwise posterior decoding selects $\\hat{q}_t = \\arg\\max_i P(q_t = S_i \\mid O)$.
            This maximizes the expected number of correct individual time steps. However, it completely ignores transition constraints $A_{ij}$.
            If $A_{ij} = 0$, pointwise argmax can propose illegal kinematic transitions.
            The <strong>Viterbi Algorithm</strong> optimizes the joint probability $P(q_1, \\dots, q_T, O \\mid \\lambda)$ over valid paths.
        </p>
    </div>
    <div style="margin-bottom: 1.25rem;">
        <h4>3. Why are log-space and scaled formulations necessary?</h4>
        <p>
            <strong>Answer:</strong> For a sequence of length $T = 100$ at $25\\text{ Hz}$, multiplying $100$ probabilities
            typically yields magnitudes of order $10^{-100}$ to $10^{-250}$, severely underflowing standard IEEE 754 64-bit floating point limit ($\\sim 10^{-308}$).
            The log-space algorithm uses the max-subtraction <code>logsumexp</code> trick to operate in logarithmic domain without underflow.
            The scaled algorithm normalizes $\\alpha_t$ at each step by $c_t = 1 / \\sum \\alpha_t(i)$, keeping values bounded between $0$ and $1$.
        </p>
    </div>
    <div style="margin-bottom: 1.25rem;">
        <h4>4. What is the algorithmic complexity of Forward and Backward?</h4>
        <p>
            <strong>Answer:</strong>
            <ul>
                <li><strong>Time Complexity:</strong> $\\mathcal{O}(T \\cdot N^2)$, where $T$ is sequence length and $N$ is the number of hidden states.
                At each of the $T-1$ steps, each of the $N$ destination states computes a dot-product over $N$ source states.</li>
                <li><strong>Space Complexity:</strong> $\\mathcal{O}(T \\cdot N)$ to store the complete intermediate $\\alpha$ and $\\beta$ trellis tables.</li>
            </ul>
        </p>
    </div>
    <div style="margin-bottom: 1.25rem;">
        <h4>5. Why use Cholesky decomposition instead of inverting Σ?</h4>
        <p>
            <strong>Answer:</strong> Explicit matrix inversion $\\Sigma^{-1}$ has $\\mathcal{O}(D^3)$ complexity and is numerically unstable
            when sensor variance is low (e.g. during stationary Sitting).
            Decomposing $\\Sigma = L L^T$ via Cholesky factor $L$ allows computing the log-determinant simply as $2 \\sum \\log L_{kk}$
            and Mahalanobis quadratic distance via forward substitution $L v = (x - \\mu)$, which is guaranteed stable and twice as fast.
        </p>
    </div>
    <div>
        <h4>6. How is temporal data leakage prevented?</h4>
        <p>
            <strong>Answer:</strong> Shuffling consecutive time steps or windows across splits destroys temporal correlation and leaks posture signatures.
            We strictly enforce <strong>participant-wise splitting</strong> (Train: 101–106, Val: 107, Test: 108–109) and Leave-One-Subject-Out cross-validation.
            The <code>SubjectScaler</code> is fit strictly on training participants and never sees test subjects.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Academic References")
st.markdown("""
1. **Rabiner, L. R. (1989).** *A tutorial on Hidden Markov Models and selected applications in speech recognition*. Proceedings of the IEEE, 77(2), 257–286.
2. **Reiss, A., & Stricker, D. (2012).** *Introducing a New Benchmarked Dataset for Activity Monitoring*. In Proceedings of the 16th International Symposium on Wearable Computers (ISWC), pp. 108–109.
3. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 13: Sequential Data (pp. 605–652). Springer.
4. **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. Chapter 17: Markov and Hidden Markov Models. MIT Press.
""")
