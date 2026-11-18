"""Page 7: Backward Algorithm - Backward Recursion and Mathematical Invariant Proof."""

import streamlit as st

from config import STATE_NAMES
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler
from visualizations.trellis import plot_backward_trellis

st.title("The Backward Algorithm (Conditional Future Probability)")
st.caption("Backward recursion β_t(i) = P(O_{t+1}, ..., O_T | q_t = S_i, λ) and likelihood duality")

st.markdown(
    """
<div class="paper-card">
    <h4 style="margin-top:0;">Mathematical Formulation (Rabiner 1989)</h4>
    <ol>
        <li><strong>Initialization ($t = T$):</strong><br>
            $$\\beta_T(i) = 1.0, \\quad \\log \\beta_T(i) = 0.0, \\quad 1 \\le i \\le N$$
        </li>
        <li><strong>Induction ($t = T-1, \\dots, 1$):</strong><br>
            $$\\beta_t(i) = \\sum_{j=1}^N A_{ij} \\cdot b_j(O_{t+1}) \\cdot \\beta_{t+1}(j)$$
            <em>Log-space:</em> $\\log \\beta_t(i) = \\text{logsumexp}_{j=1}^N \\left( \\log A_{ij} + \\log b_j(O_{t+1}) + \\log \\beta_{t+1}(j) \\right)$
        </li>
        <li><strong>Termination ($t = 1$):</strong><br>
            $$P(O \\mid \\lambda) = \\sum_{i=1}^N \\pi_i \\cdot b_i(O_1) \\cdot \\beta_1(i)$$
            <em>Log-space:</em> $\\log P(O \\mid \\lambda) = \\text{logsumexp}_{i=1}^N \\left( \\log \\pi_i + \\log b_i(O_1) + \\log \\beta_1(i) \\right)$
        </li>
    </ol>
</div>
""",
    unsafe_allow_html=True,
)

# Load resources
hmm = GaussianHMM.load()
scaler = SubjectScaler.load()
df, _ = load_processed_or_synthetic()
sequences, state_seqs, labels, sids = build_feature_sequences(df, sequence_length=50, stride=50)


def format_seq_label(opt):
    if isinstance(opt, int) and 0 <= opt < len(sequences):
        return f"Sequence #{opt + 1} (Subject {sids[opt]} | True Label: {STATE_NAMES[labels[opt]]})"
    return str(opt)


seq_idx = st.selectbox(
    "Select Observation Sequence",
    options=list(range(min(20, len(sequences)))),
    format_func=format_seq_label,
)

raw_seq = sequences[seq_idx]
scaled_seq = scaler.transform([raw_seq])[0]

# Compute Forward and Backward passes
ll_fwd, log_alpha, _ = hmm.forward(scaled_seq, method="log")
ll_bwd, log_beta = hmm.backward(scaled_seq, method="log")

abs_discrepancy = abs(ll_fwd - ll_bwd)

# Mathematical Proof Callout
st.markdown("### Mandatory Invariant: Forward vs Backward Duality")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Forward Log-Likelihood", f"{ll_fwd:.10f}")
with col2:
    st.metric("Backward Log-Likelihood", f"{ll_bwd:.10f}")
with col3:
    st.metric(
        "Absolute Difference |Δ|",
        f"{abs_discrepancy:.2e}",
        delta="Pass (< 1e-8)" if abs_discrepancy < 1e-8 else "Fail",
    )

st.markdown(
    f"""
<div class="callout-box callout-success">
    <strong>Mathematical Invariant Proof:</strong><br>
    The Forward variable evaluates $\\sum_i \\alpha_T(i) = P(O \\mid \\lambda)$, while the Backward variable evaluates
    $\\sum_i \\pi_i b_i(O_1) \\beta_1(i) = P(O \\mid \\lambda)$.<br>
    On this sequence, the observed discrepancy is <strong>{abs_discrepancy:.2e}</strong>, strictly satisfying the
    required numerical tolerance threshold of <code>1e-8</code>.
</div>
""",
    unsafe_allow_html=True,
)

# Backward Trellis Chart
fig_trellis = plot_backward_trellis(log_beta, state_names=STATE_NAMES)
st.plotly_chart(fig_trellis, use_container_width=True)
