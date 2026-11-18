"""Page 4: Sensor Dynamics - Interactive 3D Trajectory and Multi-Axis Signals."""

import streamlit as st

from config import STATE_NAMES
from preprocessing.loader import load_processed_or_synthetic
from visualizations.charts import plot_3d_trajectory, plot_triaxial_signals

st.title("Sensor Dynamics & Kinematic Visualisation")
st.caption("Interactive multi-axis accelerometer waveforms and 3D phase-space trajectories")

# Load cached data
df, is_synthetic = load_processed_or_synthetic()

# Sequence and Subject Filters
col_sub, col_act = st.columns(2)
with col_sub:
    subjects = sorted(list(df["subject_id"].unique())) if "subject_id" in df.columns else [101]
    selected_subject = st.selectbox("Select Subject", options=subjects, index=0)

with col_act:
    available_acts = (
        sorted(list(df["activity_name"].unique())) if "activity_name" in df.columns else STATE_NAMES
    )
    selected_activity = st.selectbox("Select Activity", options=available_acts, index=0)

filtered_df = df[
    (df["subject_id"] == selected_subject) & (df["activity_name"] == selected_activity)
].reset_index(drop=True)

if filtered_df.empty:
    filtered_df = df.head(300)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(filtered_df))
with col2:
    st.metric("Duration", f"{len(filtered_df) / 25.0:.1f} s")
with col3:
    mean_mag = float(filtered_df["acc_mag"].mean()) if "acc_mag" in filtered_df.columns else 9.81
    st.metric("Mean Magnitude", f"{mean_mag:.2f} m/s²")
with col4:
    std_mag = float(filtered_df["acc_mag"].std()) if "acc_mag" in filtered_df.columns else 0.5
    st.metric("Dynamic Variance (std)", f"{std_mag:.2f} m/s²")

# Charts
st.markdown("### Accelerometer Channels (ax, ay, az) and Resultant Magnitude")
fig_signals = plot_triaxial_signals(filtered_df, max_samples=400)
st.plotly_chart(fig_signals, use_container_width=True)

st.markdown("### 3D Kinematic Acceleration Space")
fig_3d = plot_3d_trajectory(filtered_df, max_points=350)
st.plotly_chart(fig_3d, use_container_width=True)
