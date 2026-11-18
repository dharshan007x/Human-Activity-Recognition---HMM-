"""Page 8: Posterior Probabilities - Smoothing and Dynamic Time-Step Slider Animation."""

import streamlit as st

from config import STATE_NAMES
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler
from visualizations.trellis import (
    plot_posterior_trellis,
    plot_step_probabilities_bars,
)

st.title("Posterior State Probability Smoothing")
st.caption(
    "Combining past evidence (α) and future evidence (β) to compute γ_t(i) = P(q_t = S_i | O, λ)"
)

st.markdown(
    """
<div class="paper-card">
    <h4 style="margin-top:0;">Smoothing Formulation & Theoretical Distinction</h4>
    <p>
        The posterior probability of being in state $S_i$ at time step $t$, given the entire observation sequence $O$:
        $$\\gamma_t(i) = \\frac{\\alpha_t(i) \\cdot \\beta_t(i)}{P(O \\mid \\lambda)} = \\frac{\\alpha_t(i) \\beta_t(i)}{\\sum_{j=1}^N \\alpha_t(j) \\beta_t(j)}$$
        <em>Log-space:</em> $\\log \\gamma_t(i) = \\log \\alpha_t(i) + \\log \\beta_t(i) - \\log P(O \\mid \\lambda)$
    </p>
    <div class="callout-box callout-warning">
        <strong>Vital Defense Principle: Pointwise Argmax vs Viterbi Path</strong><br>
        The sequence of pointwise maximum posteriors $\\hat{q}_t = \\arg\\max_i \\gamma_t(i)$ selects the <em>individually most likely state</em> at each isolated time instant.
        It does <strong>not</strong> optimize the joint path probability $P(q_1, \\dots, q_T \\mid O)$ and can propose physically impossible transitions where $A_{ij} = 0$.
        The globally optimal legal state sequence is found exclusively by the <strong>Viterbi Algorithm</strong> (see Beyond the Core page).
    </div>
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

# Compute Posteriors
gamma, states, ll, log_alpha, log_beta = hmm.posterior(scaled_seq, method="log")

# Trellis visualization
fig_gamma = plot_posterior_trellis(gamma, state_names=STATE_NAMES)
st.plotly_chart(fig_gamma, use_container_width=True)

st.markdown("### Interactive Time-Step Slider: State Probability Animation")
st.markdown(
    "Drag the slider below to step through time $t$ and inspect the exact distribution of $\\gamma_t(i)$ across states."
)

step_t = st.slider("Time Step t", min_value=1, max_value=len(scaled_seq), value=1)
t_idx = step_t - 1

col_chart, col_details = st.columns([1.5, 1])

with col_chart:
    fig_bars = plot_step_probabilities_bars(
        gamma[t_idx], step_index=step_t, var_name="γ", state_names=STATE_NAMES
    )
    st.plotly_chart(fig_bars, use_container_width=True)

with col_details:
    st.markdown(f"#### Time Step t = {step_t} Probabilities")
    for i, name in enumerate(STATE_NAMES):
        prob = gamma[t_idx, i]
        st.write(f"- **{name}:** `{prob * 100:.2f}%`")

    predicted_state = STATE_NAMES[states[t_idx]]
    st.markdown(
        f"""
    <div class="callout-box callout-success">
        <strong>Pointwise Decoded State:</strong><br>
        <span style="font-size: 1.2rem; font-weight: 600;">{predicted_state}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
