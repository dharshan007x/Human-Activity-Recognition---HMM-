"""Page 5: HMM Architecture - Mathematical Parameters, Pi, A, and Cholesky Gaussians."""

import numpy as np
import pandas as pd
import streamlit as st

from config import STATE_NAMES
from models.hmm_model import GaussianHMM
from visualizations.charts import plot_transition_matrix

st.title("Gaussian Hidden Markov Model Architecture")
st.caption("Supervised MLE parameterization, regularized covariance, and Cholesky log-densities")

# Load trained model
hmm = GaussianHMM.load()

st.markdown(
    """
<div class="paper-card">
    <h4 style="margin-top:0;">Model Definition λ = (π, A, B)</h4>
    <p>
        The system models physical activity as a 3-state first-order Hidden Markov Model with continuous 4D observations:
        $$O_t = [a_x, a_y, a_z, a_{\\text{mag}}]^T \\in \\mathbb{R}^4$$
    </p>
    <ul>
        <li><strong>Initial State Distribution π:</strong> Probabilities of starting in each activity state (with Laplace smoothing).</li>
        <li><strong>Transition Matrix A:</strong> Transition dynamics $A_{ij} = P(q_{t+1} = S_j \\mid q_t = S_i)$.</li>
        <li><strong>Emission Distribution B:</strong> Multivariate Gaussian density $\\mathcal{N}(\\mu_j, \\Sigma_j)$ with regularized covariance $\\Sigma_j + \\epsilon I$.</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Initial State Distribution π")
    pi_df = pd.DataFrame(
        {
            "Activity State": STATE_NAMES,
            "Probability π_i": np.round(hmm.pi, 4),
            "Percentage": [f"{p * 100:.2f}%" for p in hmm.pi],
        }
    )
    st.dataframe(pi_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Transition Probability Matrix A")
    fig_A = plot_transition_matrix(hmm.A, state_names=STATE_NAMES)
    st.plotly_chart(fig_A, use_container_width=True)

st.markdown("### Multivariate Gaussian Emission Density Parameters")
st.markdown(
    """
<div class="callout-box">
    <strong>Cholesky Factorization for Inversion Stability:</strong><br>
    Direct matrix inversion of covariance $\\Sigma_j^{-1}$ is prone to numerical instability during static sitting.
    We enforce positive definiteness via regularizing jitter ($\\Sigma_j + 10^{-4} I$) and evaluate emissions in log space
    using the lower-triangular Cholesky factor $L_j$:
    $$\\log b_j(O_t) = -\\frac{D}{2}\\log(2\\pi) - \\sum_{k=1}^D \\log (L_j)_{kk} - \\frac{1}{2} \\|L_j^{-1}(O_t - \\mu_j)\\|^2$$
</div>
""",
    unsafe_allow_html=True,
)

# State Means and Variances Table
emission_rows = []
for idx, name in enumerate(STATE_NAMES):
    em = hmm.emissions[idx]
    mean_str = ", ".join([f"{v:.2f}" for v in em.mean])
    var_str = ", ".join([f"{v:.2f}" for v in np.diag(em.cov)])
    emission_rows.append(
        {
            "Activity": name,
            "Mean Vector [ax, ay, az, amag]": f"[{mean_str}]",
            "Feature Variances [σ²x, σ²y, σ²z, σ²mag]": f"[{var_str}]",
            "Log-Determinant log|Σ|": f"{em.log_det_cov:.3f}",
        }
    )

emission_df = pd.DataFrame(emission_rows)
st.dataframe(emission_df, use_container_width=True, hide_index=True)
