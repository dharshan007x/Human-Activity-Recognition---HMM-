"""Page 9: Activity Recognition - Live Sequence Prediction and CSV Upload with Schema Check."""

import numpy as np
import pandas as pd
import streamlit as st

from config import FEATURE_NAMES, MAX_PHYSICAL_ACCEL_MS2, STATE_NAMES
from models.hmm_model import GaussianHMM
from preprocessing.features import build_feature_sequences, extract_features
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler
from visualizations.trellis import plot_posterior_trellis

st.title("Human Activity Recognition & Inference Tool")
st.caption(
    "Classify sequential accelerometer movements using Forward evaluation and Posterior decoding"
)

# Load model and scaler
hmm = GaussianHMM.load()
scaler = SubjectScaler.load()

# Tab 1: Built-in Benchmark Sequences, Tab 2: Custom CSV Upload
tab1, tab2 = st.tabs(["📁 Select Benchmark Sequence", "📤 Upload Custom Accelerometer CSV"])

selected_features = None
display_name = ""

with tab1:
    df, _ = load_processed_or_synthetic()
    sequences, state_seqs, labels, sids = build_feature_sequences(df, sequence_length=50, stride=50)

    def format_seq_label(opt):
        if isinstance(opt, int) and 0 <= opt < len(sequences):
            return f"Test Seq #{opt + 1} | Participant {sids[opt]} | True Label: {STATE_NAMES[labels[opt]]}"
        return str(opt)

    seq_idx = st.selectbox(
        "Choose a sequence from held-out test participants",
        options=list(range(min(25, len(sequences)))),
        format_func=format_seq_label,
    )
    raw_seq = sequences[seq_idx]
    display_name = (
        f"Benchmark Sequence #{seq_idx + 1} (Ground Truth: {STATE_NAMES[labels[seq_idx]]})"
    )
    selected_features = raw_seq

with tab2:
    st.markdown(
        """
    <div class="paper-card">
        <h4 style="margin-top:0;">Required CSV Format Specification</h4>
        <p>The uploaded CSV must contain columns: <code>acc_x</code>, <code>acc_y</code>, <code>acc_z</code> (acceleration in m/s²).
        Optional: <code>timestamp</code>. Physical values must be within ±16g (±156.96 m/s²).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Downloadable CSV Template
    template_df = pd.DataFrame(
        {
            "timestamp": [0.00, 0.04, 0.08, 0.12, 0.16],
            "acc_x": [0.45, 0.52, 0.38, 0.49, 0.42],
            "acc_y": [9.75, 9.82, 9.68, 9.77, 9.71],
            "acc_z": [1.12, 1.08, 1.15, 1.10, 1.14],
        }
    )
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV Template",
        data=csv_bytes,
        file_name="accelerometer_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            # Validation 1: Required columns
            req_cols = ["acc_x", "acc_y", "acc_z"]
            missing = [c for c in req_cols if c not in user_df.columns]
            if missing:
                st.error(f"❌ Schema Validation Failed: Missing required columns: {missing}")
            elif len(user_df) < 5:
                st.error("❌ Schema Validation Failed: CSV contains fewer than 5 time steps.")
            else:
                # Validation 2: Physical bounds check
                unphysical = (user_df[req_cols].abs() > MAX_PHYSICAL_ACCEL_MS2).any()
                if unphysical.any():
                    st.warning(
                        f"⚠️ Physical Outlier Warning: Values exceed ±16g ({MAX_PHYSICAL_ACCEL_MS2} m/s²). Will be clipped."
                    )

                user_feat = extract_features(user_df)
                selected_features = user_feat[FEATURE_NAMES].to_numpy(dtype=np.float64)
                display_name = f"Uploaded File: {uploaded_file.name} ({len(user_df)} samples)"
                st.success(f"✅ CSV Loaded Successfully: {len(user_df)} time steps parsed.")
        except Exception as e:
            st.error(f"❌ Error parsing CSV file: {e}")

# Run Inference if sequence is ready
if selected_features is not None and len(selected_features) > 0:
    st.markdown(f"### Inference Results: {display_name}")

    scaled_seq = scaler.transform([selected_features])[0]
    gamma, states, ll, log_alpha, log_beta = hmm.posterior(scaled_seq, method="log")
    viterbi_states, viterbi_ll, _ = hmm.decode_viterbi(scaled_seq)

    # Dominant state
    state_counts = np.bincount(states, minlength=len(STATE_NAMES))
    dominant_state_idx = int(np.argmax(state_counts))
    dominant_state_name = STATE_NAMES[dominant_state_idx]
    mean_confidence = float(np.mean(np.max(gamma, axis=1)))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Primary Activity", dominant_state_name)
    with col2:
        st.metric("Mean Posterior Confidence", f"{mean_confidence * 100:.1f}%")
    with col3:
        st.metric("Sequence Log-Likelihood", f"{ll:.2f}")
    with col4:
        st.metric("Per-Step Mean NLL", f"{-ll / len(scaled_seq):.2f}")

    # Plot posterior trellis
    fig_gamma = plot_posterior_trellis(
        gamma, state_names=STATE_NAMES, title=f"State Posteriors over Time: {dominant_state_name}"
    )
    st.plotly_chart(fig_gamma, use_container_width=True)

    # Step-by-step table
    results_df = pd.DataFrame(
        {
            "Time Step": np.arange(1, len(scaled_seq) + 1),
            "Pointwise State": [STATE_NAMES[s] for s in states],
            "Viterbi State": [STATE_NAMES[s] for s in viterbi_states],
            "P(Walking)": np.round(gamma[:, 0], 4),
            "P(Sitting)": np.round(gamma[:, 1], 4),
            "P(Running)": np.round(gamma[:, 2], 4),
            "Confidence": np.round(np.max(gamma, axis=1), 4),
        }
    )

    with st.expander("📋 View and Download Step-by-Step Prediction Table"):
        st.dataframe(results_df, use_container_width=True)
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Inference CSV",
            data=csv_out,
            file_name="inference_predictions.csv",
            mime="text/csv",
        )
