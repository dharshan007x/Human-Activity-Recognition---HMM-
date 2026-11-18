"""Page 6: Forward Algorithm - Mathematical Induction, Trellis, and Likelihood Convergence."""

import numpy as np
import streamlit as st

from config import STATE_NAMES
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler
from visualizations.trellis import plot_forward_trellis

st.title("The Forward Algorithm (Evaluation & Filtering)")
st.caption(
    "Recursive computation of joint observation-state likelihoods α_t(i) = P(O_1, ..., O_t, q_t = S_i | λ)"
)

st.markdown(
    """
<div class="paper-card">
    <h4 style="margin-top:0;">Mathematical Formulation (Rabiner 1989)</h4>
    <ol>
        <li><strong>Initialization ($t = 1$):</strong><br>
            $$\\alpha_1(i) = \\pi_i \\cdot b_i(O_1), \\quad 1 \\le i \\le N$$
            <em>Log-space:</em> $\\log \\alpha_1(i) = \\log \\pi_i + \\log b_i(O_1)$
        </li>
        <li><strong>Induction ($t = 1, \\dots, T-1$):</strong><br>
            $$\\alpha_{t+1}(j) = \\left[ \\sum_{i=1}^N \\alpha_t(i) A_{ij} \\right] b_j(O_{t+1}), \\quad 1 \\le j \\le N$$
            <em>Log-space:</em> $\\log \\alpha_{t+1}(j) = \\text{logsumexp}_{i=1}^N \\left( \\log \\alpha_t(i) + \\log A_{ij} \\right) + \\log b_j(O_{t+1})$
        </li>
        <li><strong>Termination:</strong><br>
            $$P(O \\mid \\lambda) = \\sum_{i=1}^N \\alpha_T(i), \\quad \\log P(O \\mid \\lambda) = \\text{logsumexp}_{i=1}^N \\left( \\log \\alpha_T(i) \\right)$$
        </li>
    </ol>
</div>
""",
    unsafe_allow_html=True,
)

# Load model and sequences
hmm = GaussianHMM.load()
scaler = SubjectScaler.load()
df, _ = load_processed_or_synthetic()
sequences, state_seqs, labels, sids = build_feature_sequences(df, sequence_length=50, stride=50)

# Interactive Sequence Selector
col_sel, col_algo = st.columns(2)
with col_sel:

    def format_seq_label(opt):
        if isinstance(opt, int) and 0 <= opt < len(sequences):
            return f"Sequence #{opt + 1} (Subject {sids[opt]} | True Label: {STATE_NAMES[labels[opt]]})"
        return str(opt)

    seq_idx = st.selectbox(
        "Select Test Observation Sequence",
        options=list(range(min(20, len(sequences)))),
        format_func=format_seq_label,
    )
with col_algo:
    calc_mode = st.radio(
        "Numerical Formulation",
        options=["Log-Space (logsumexp)", "Scaled (Rabiner c_t)"],
        horizontal=True,
    )

raw_seq = sequences[seq_idx]
scaled_seq = scaler.transform([raw_seq])[0]

# Compute Forward Pass
method = "log" if "Log" in calc_mode else "scaled"
ll, alpha_table, c_scale = hmm.forward(scaled_seq, method=method)

# Metrics display
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Sequence Length T", f"{len(scaled_seq)} steps ({len(scaled_seq) / 25.0:.1f} s)")
with col_m2:
    st.metric("Sequence Log-Likelihood", f"{ll:.4f}")
with col_m3:
    st.metric("Mean NLL per step", f"{-ll / len(scaled_seq):.4f}")

# Trellis visualization
if method == "log":
    fig_trellis = plot_forward_trellis(alpha_table, state_names=STATE_NAMES)
else:
    fig_trellis = plot_forward_trellis(
        np.log(np.maximum(1e-12, alpha_table)),
        state_names=STATE_NAMES,
        title="Scaled Forward Trellis log(α_hat)",
    )
st.plotly_chart(fig_trellis, use_container_width=True)

# Step-by-Step Numerical Calculation Expander
with st.expander("🔬 Show the Step-by-Step Calculation (Real Numbers)"):
    step_t = st.slider(
        "Select time step t for detailed inspection",
        min_value=1,
        max_value=len(scaled_seq),
        value=1,
    )
    t_idx = step_t - 1

    st.write(f"**Inspection at step t = {step_t}:**")
    st.write(f"- Observation vector $O_{step_t}$: `{np.round(raw_seq[t_idx], 3)}`")
    st.write(f"- Standardized vector $z_{step_t}$: `{np.round(scaled_seq[t_idx], 3)}`")

    log_em = hmm.compute_log_emissions(scaled_seq[t_idx : t_idx + 1])[0]
    for i, name in enumerate(STATE_NAMES):
        st.write(
            f"  * State **{name}**: $\\log b({name}) = {log_em[i]:.4f}$, $\\log \\alpha_{step_t}({name}) = {alpha_table[t_idx, i]:.4f}$"
        )
