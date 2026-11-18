"""Page 1: Home - Executive Briefing and Architecture Overview."""

import streamlit as st

st.title("Human Activity Recognition via Forward & Backward Algorithms")
st.caption(
    "A mathematically defensible, university-grade benchmark on triaxial accelerometer sequences"
)

st.markdown(
    """
<div class="paper-card">
    <h3 style="margin-top:0;">Official Problem Statement</h3>
    <p>
        Identify human activities such as <strong>Walking</strong>, <strong>Sitting</strong>, and <strong>Running</strong>
        from sequential accelerometer readings. Model activities as hidden states and motion dynamics as continuous
        multivariate Gaussian observations, then use the <strong>Forward</strong> and <strong>Backward Algorithms</strong>
        to calculate sequence likelihoods and posterior state probabilities.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Hidden States", "3", help="Walking, Sitting, Running")
with col2:
    st.metric("Sensor Placement", "Chest IMU", help="±16g Accelerometer (Colibri Wireless)")
with col3:
    st.metric("Decimated Rate", "25 Hz", help="Downsampled from 100 Hz via 4th-order Butterworth")
with col4:
    st.metric("Core Engine", "Pure NumPy", help="Zero 3rd-party HMM libraries in core inference")

st.markdown("### System Pipeline Architecture")
st.markdown("""
```
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
 │ PAMAP2 Raw Sensors   │ ───> │ Anti-Aliasing & Decimate│ ───> │ Feature Construction  │
 │ (100 Hz, ±16g Chest) │      │ (10 Hz Cutoff -> 25 Hz) │      │ [ax, ay, az, a_mag]   │
 └──────────────────────┘      └─────────────────────────┘      └───────────────────────┘
                                                                            │
 ┌──────────────────────────────────────────────────────────────────────────┘
 │
 ▼
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
 │ Subject Partitioning │ ───> │ Supervised Fit (MLE)    │ ───> │ Forward & Backward    │
 │ (Zero Leakage Split) │      │ (Pi, A, Cholesky Gauss) │      │ Inference Algorithms  │
 └──────────────────────┘      └─────────────────────────┘      └───────────────────────┘
                                                                            │
 ┌──────────────────────────────────────────────────────────────────────────┘
 │
 ▼
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
 │ Posterior Smoothing  │ ───> │ Comprehensive Benchmark │ ───> │ Interactive Dashboard │
 │ Gamma_t = Alpha*Beta │      │ (LOSO, ECE, Bootstrap)  │      │ & CSV Inference Tool  │
 └──────────────────────┘      └─────────────────────────┘      └───────────────────────┘
```
""")

st.markdown(
    """
<div class="paper-card">
    <h4>Academic Defense Principles</h4>
    <ul>
        <li><strong>First-Principles Implementation:</strong> The Forward, Backward, Posterior, and Viterbi algorithms are written from scratch in pure NumPy.</li>
        <li><strong>Strict Verification:</strong> Forward and Backward sequence likelihoods agree to within <code>1e-8</code>. Tested against analytical hand-calculated benchmarks and the textbook Weather HMM.</li>
        <li><strong>Independent Oracle:</strong> <code>hmmlearn</code> is used exclusively as an independent verification oracle in tests.</li>
        <li><strong>Zero Data Leakage:</strong> Training, validation, and testing partitions are segmented strictly by participant ID. Scaler parameters are fit only on training participants.</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
