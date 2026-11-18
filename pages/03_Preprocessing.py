"""Page 3: Preprocessing - Signal Conditioning, Anti-Aliasing, and Zero-Leakage Splitting."""

import streamlit as st

from visualizations.charts import plot_filter_frequency_response

st.title("Signal Conditioning & Preprocessing Pipeline")
st.caption("Anti-aliasing decimation, packet loss handling, and subject isolation protocol")

st.markdown(
    """
<div class="paper-card">
    <h4 style="margin-top:0;">Anti-Aliasing Filter & Decimation Rationale (100 Hz → 25 Hz)</h4>
    <p>
        Human ambulation resides in low frequencies: walking cadence occurs at ~1.6–2.0 Hz,
        running stride frequency at ~2.6–3.2 Hz, with higher kinematic harmonics dissipating above 8–10 Hz.
        Downsampling to <strong>25 Hz</strong> (Nyquist limit = 12.5 Hz) without pre-filtering causes high-frequency sensor noise
        to fold (alias) into the human locomotion band.
    </p>
    <p>
        To prevent aliasing, signals are conditioned with a <strong>4th-order zero-phase digital Butterworth low-pass filter</strong>
        with a cutoff frequency of <strong>10.0 Hz</strong>. Forward-backward filtering (<code>scipy.signal.filtfilt</code>) guarantees
        exactly <strong>zero phase delay</strong>, preserving true impact peak timing.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Frequency response chart
fig_filter = plot_filter_frequency_response(cutoff_hz=10.0, fs=100.0, order=4)
st.plotly_chart(fig_filter, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
    <div class="paper-card">
        <h4>Time-Aware Missing Value Handling</h4>
        <ul>
            <li><strong>Bluetooth Packet Loss:</strong> Wireless IMUs occasionally drop 1–3 consecutive samples.</li>
            <li><strong>Intra-Segment Interpolation:</strong> Short bursts (&le; 12 samples = 0.5 s at 25 Hz) are imputed via linear interpolation.</li>
            <li><strong>Strict Zero Leakage:</strong> Interpolation is strictly confined within single-subject and single-activity blocks. <em>Never interpolate across participant boundaries or activity transitions!</em></li>
            <li><strong>Large Gap Protocol:</strong> Gaps &gt; 0.5 s are partitioned into separate continuous sequences.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="paper-card">
        <h4>Subject-Wise Splitting & Scaler Isolation</h4>
        <ul>
            <li><strong>The Flaw of Random Splitting:</strong> Shuffling temporal sequences leaks posture signatures across splits.</li>
            <li><strong>Participant Partitioning:</strong>
                <ul>
                    <li><strong>Training:</strong> Subjects 101, 102, 103, 104, 105, 106</li>
                    <li><strong>Validation:</strong> Subject 107</li>
                    <li><strong>Held-out Test:</strong> Subjects 108, 109</li>
                </ul>
            </li>
            <li><strong>Isolated Normalization:</strong> <code>SubjectScaler</code> is fit strictly on training participants, then serialized into <code>scaler.json</code> and applied during inference.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
