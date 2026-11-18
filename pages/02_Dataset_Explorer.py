"""Page 2: Dataset Explorer - PAMAP2 Benchmark Attributes and Demographics."""

import pandas as pd
import streamlit as st

from config import PAMAP2_LICENCE, PAMAP2_SOURCE_URL

st.title("PAMAP2 Benchmark Dataset Explorer")
st.caption("UCI Machine Learning Repository Benchmark for Physical Activity Monitoring")

st.markdown(
    f"""
<div class="paper-card">
    <h4 style="margin-top:0;">Dataset Provenance & Metadata</h4>
    <p><strong>Repository:</strong> UCI Machine Learning Repository (Dataset ID 231)</p>
    <p><strong>Licence:</strong> {PAMAP2_LICENCE}</p>
    <p><strong>Canonical Source:</strong> <a href="{PAMAP2_SOURCE_URL}" target="_blank">Download Archive</a></p>
    <p><strong>Citation:</strong> Reiss, A., & Stricker, D. (2012). <em>Introducing a New Benchmarked Dataset for Activity Monitoring</em>. In Proc. 16th IEEE ISWC, pp. 108–109.</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Participant Demographics (9 Subjects)")
    demographics = pd.DataFrame(
        {
            "Attribute": [
                "Total Subjects",
                "Gender Distribution",
                "Mean Age",
                "Mean Height",
                "Mean Weight",
                "Mean BMI",
                "Dominant Arm",
            ],
            "Value": [
                "9 participants",
                "8 Male, 1 Female",
                "27.2 ± 3.3 years",
                "179.6 ± 5.1 cm",
                "80.9 ± 8.8 kg",
                "25.1 ± 2.6 kg/m²",
                "8 Right, 1 Left",
            ],
        }
    )
    st.dataframe(demographics, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Primary Activity State Mapping")
    activity_table = pd.DataFrame(
        [
            {
                "State Name": "Sitting",
                "PAMAP2 ID": 2,
                "Sensory Morphology": "Static gravitational vector, minimal dynamic variance",
            },
            {
                "State Name": "Walking",
                "PAMAP2 ID": 4,
                "Sensory Morphology": "Periodic gait cycle (1.6-2.0 Hz), torso oscillation",
            },
            {
                "State Name": "Running",
                "PAMAP2 ID": 5,
                "Sensory Morphology": "Ballistic impact spikes (2.6-3.2 Hz), high magnitude",
            },
            {
                "State Name": "Transient (Dropped)",
                "PAMAP2 ID": 0,
                "Sensory Morphology": "Non-steady transitions between protocols",
            },
        ]
    )
    st.dataframe(activity_table, use_container_width=True, hide_index=True)

st.markdown("### Sensor Channel Layout & ±16g Justification")
st.markdown(
    """
<div class="callout-box">
    <strong>Why the ±16g Sensor is Required:</strong><br>
    Each Colibri IMU records from two 3D accelerometers: a high-resolution ±6g sensor and an extended-range ±16g sensor.
    During ground-strike impacts in Running, torso and extremity shock waves regularly exceed 6g (peaking between 8g and 12g).
    The ±6g sensor saturates and clips, creating severe non-Gaussian plateaus. The ±16g sensor preserves the natural physical signal shape.
</div>
""",
    unsafe_allow_html=True,
)

channel_layout = pd.DataFrame(
    {
        "Column Indices": ["Col 0", "Col 1", "Col 2", "Cols 3–19", "Cols 20–36", "Cols 37–53"],
        "Subsystem": [
            "Timestamp",
            "Activity ID",
            "Heart Rate Monitor",
            "Hand IMU (Dominant Wrist)",
            "Chest IMU (Sternum)",
            "Ankle IMU (Dominant Leg)",
        ],
        "Details": [
            "Seconds from start",
            "Integer activity identifier",
            "~9 Hz sampling (~91% NaNs in 100 Hz grid)",
            "3D acc (±16g & ±6g), gyro, mag, orientation",
            "3D acc (±16g: Cols 21–23), gyro, mag",
            "3D acc (±16g: Cols 38–40), gyro, mag",
        ],
    }
)
st.dataframe(channel_layout, use_container_width=True, hide_index=True)
