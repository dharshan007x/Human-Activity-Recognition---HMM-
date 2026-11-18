"""Page 11: Beyond the Core - Viterbi Decoding and Filtering vs Smoothing (Extensions)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ACTIVITY_COLORS, PALETTE, STATE_NAMES
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler
from visualizations.theme import apply_warm_layout

st.title("Beyond the Core (Advanced Additions)")
st.caption(
    "Advanced HMM topics: Globally optimal Viterbi paths, Filtering vs Smoothing, and Baum-Welch re-estimation"
)

st.markdown(
    """
<div class="callout-box callout-warning">
    <strong>Academic Disclaimer:</strong><br>
    The topics presented on this page are clearly designated as <strong>advanced extensions</strong>.
    They demonstrate breadth beyond the primary problem statement without replacing the central focus on Forward and Backward algorithms.
</div>
""",
    unsafe_allow_html=True,
)

# Load model and sequences
hmm = GaussianHMM.load()
scaler = SubjectScaler.load()
df, _ = load_processed_or_synthetic()
sequences, state_seqs, labels, sids = build_feature_sequences(df, sequence_length=50, stride=50)

# Section 1: Viterbi vs Pointwise Posterior Decoding
st.markdown("### 1. Viterbi Decoding: Globally Optimal State Sequence")
st.markdown(
    """
<div class="paper-card">
    <p>
        While posterior decoding maximizes the probability of each individual state $\\hat{q}_t = \\arg\\max_i \\gamma_t(i)$,
        the <strong>Viterbi Algorithm</strong> uses dynamic programming in log-space to find the single sequence
        $q^* = (q_1^*, \\dots, q_T^*)$ that maximizes the joint path probability:
        $$q^* = \\arg\\max_{q_1, \\dots, q_T} P(q_1, \\dots, q_T, O \\mid \\lambda)$$
    </p>
    <ul>
        <li><strong>Recursion:</strong> $V_t(j) = \\max_{i} \\left[ V_{t-1}(i) + \\log A_{ij} \\right] + \\log b_j(O_t)$</li>
        <li><strong>Backpointers:</strong> $B_t(j) = \\arg\\max_i \\left[ V_{t-1}(i) + \\log A_{ij} \\right]$</li>
        <li><strong>Guarantee:</strong> Viterbi strictly adheres to transition topology and cannot propose illegal transitions ($A_{ij} = 0$).</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)


def format_seq_label(opt):
    if isinstance(opt, int) and 0 <= opt < len(sequences):
        return f"Sequence #{opt + 1} (True Label: {STATE_NAMES[labels[opt]]})"
    return str(opt)


seq_idx = st.selectbox(
    "Select sequence to compare Pointwise vs Viterbi paths",
    options=list(range(min(15, len(sequences)))),
    format_func=format_seq_label,
)

raw_seq = sequences[seq_idx]
scaled_seq = scaler.transform([raw_seq])[0]

# Run both decoders
gamma, pointwise_states, _, _, _ = hmm.posterior(scaled_seq, method="log")
viterbi_states, viterbi_ll, _ = hmm.decode_viterbi(scaled_seq)

# Compare Paths
comparison_df = pd.DataFrame(
    {
        "Time Step": np.arange(1, len(scaled_seq) + 1),
        "Pointwise Smoothing State": [STATE_NAMES[s] for s in pointwise_states],
        "Viterbi Optimal Path State": [STATE_NAMES[s] for s in viterbi_states],
        "Pointwise Confidence": np.round(np.max(gamma, axis=1), 4),
        "Path Agreement": [
            "Identical" if p == v else "Divergent" for p, v in zip(pointwise_states, viterbi_states)
        ],
    }
)

agreement_pct = (pointwise_states == viterbi_states).mean() * 100
col1, col2 = st.columns(2)
with col1:
    st.metric("Path Agreement", f"{agreement_pct:.1f}%")
with col2:
    st.metric("Viterbi Best Path Log-Probability", f"{viterbi_ll:.4f}")

with st.expander("🔍 View Trajectory Comparison Table"):
    st.dataframe(comparison_df, use_container_width=True)

# Section 2: Filtering vs Smoothing Comparison
st.markdown("### 2. Filtering (Alpha-only) vs Smoothing (Alpha + Beta)")
st.markdown(
    """
<div class="paper-card">
    <p>
        In real-time streaming applications, future observations $O_{t+1:T}$ are not yet available:
    </p>
    <ul>
        <li><strong>Filtered State Probability (Online):</strong> Conditioned only on past and present observations:<br>
            $$P(q_t = S_i \\mid O_{1:t}, \\lambda) = \\frac{\\alpha_t(i)}{\\sum_{j=1}^N \\alpha_t(j)}$$
        </li>
        <li><strong>Smoothed State Probability (Offline):</strong> Conditioned on the complete sequence:<br>
            $$P(q_t = S_i \\mid O_{1:T}, \\lambda) = \\gamma_t(i) = \\frac{\\alpha_t(i) \\beta_t(i)}{P(O \\mid \\lambda)}$$
        </li>
    </ul>
    <p>
        Smoothing incorporates future kinematic evidence, reducing estimation uncertainty and eliminating transient spikes.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Compute filtered probabilities
_, log_alpha, _ = hmm.forward(scaled_seq, method="log")
alpha_exp = np.exp(log_alpha - np.max(log_alpha, axis=1, keepdims=True))
filtered_probs = alpha_exp / np.sum(alpha_exp, axis=1, keepdims=True)

fig_filt_smooth = go.Figure()
time_steps = np.arange(1, len(scaled_seq) + 1)
target_state_idx = labels[seq_idx]
target_name = STATE_NAMES[target_state_idx]

fig_filt_smooth.add_trace(
    go.Scatter(
        x=time_steps,
        y=filtered_probs[:, target_state_idx],
        mode="lines",
        name=f"Filtered P({target_name} | O_1:t) [Online]",
        line=dict(color=PALETTE["neutral_data"], dash="dash", width=2),
    )
)
fig_filt_smooth.add_trace(
    go.Scatter(
        x=time_steps,
        y=gamma[:, target_state_idx],
        mode="lines",
        name=f"Smoothed γ_t({target_name}) [Offline Full Sequence]",
        line=dict(color=ACTIVITY_COLORS.get(target_name, PALETTE["accent"]), width=2.5),
    )
)

apply_warm_layout(
    fig_filt_smooth,
    title=f"Filtering vs Smoothing Trajectory for True State '{target_name}'",
    xaxis_title="Time Step t",
    yaxis_title="State Probability",
    height=360,
)
fig_filt_smooth.update_yaxes(range=[-0.05, 1.05])
st.plotly_chart(fig_filt_smooth, use_container_width=True)
