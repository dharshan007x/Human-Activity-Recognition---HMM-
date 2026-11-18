"""High-level Plotly chart generators using the field_notebook warm theme."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal

from config import (
    ACTIVITY_COLORS,
    BUTTERWORTH_CUTOFF_HZ,
    BUTTERWORTH_ORDER,
    ORIGINAL_SAMPLING_RATE_HZ,
    PALETTE,
    STATE_NAMES,
)
from visualizations.theme import apply_warm_layout


def plot_triaxial_signals(
    df: pd.DataFrame,
    title: str = "Triaxial Accelerometer Dynamics & Euclidean Magnitude",
    time_col: str = "timestamp",
    max_samples: int = 500,
) -> go.Figure:
    """Plot raw tri-axial accelerometer channels (ax, ay, az) and resultant magnitude."""
    plot_df = df.head(max_samples)
    t = plot_df[time_col].to_numpy() if time_col in plot_df.columns else np.arange(len(plot_df))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=plot_df["acc_x"],
            mode="lines",
            name="a_x (lateral)",
            line=dict(color="#8C5E58", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=plot_df["acc_y"],
            mode="lines",
            name="a_y (vertical)",
            line=dict(color=PALETTE["neutral_data"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=plot_df["acc_z"],
            mode="lines",
            name="a_z (anterior)",
            line=dict(color="#B38D48", width=1.5),
        )
    )

    if "acc_mag" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=plot_df["acc_mag"],
                mode="lines",
                name="a_mag (resultant)",
                line=dict(color=PALETTE["accent"], width=2.2, dash="dash"),
            )
        )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Acceleration (m/s²)",
        height=380,
    )
    return fig


def plot_3d_trajectory(
    df: pd.DataFrame,
    max_points: int = 400,
    title: str = "3D Kinematic Accelerometer Trajectory",
) -> go.Figure:
    """Plot 3D acceleration vector locus [ax, ay, az] colored by activity state."""
    sample = df.head(max_points)

    color_col = sample["activity_name"] if "activity_name" in sample.columns else None

    color_discrete_map = {
        "Walking": ACTIVITY_COLORS["Walking"],
        "Sitting": ACTIVITY_COLORS["Sitting"],
        "Running": ACTIVITY_COLORS["Running"],
    }

    fig = px.scatter_3d(
        sample,
        x="acc_x",
        y="acc_y",
        z="acc_z",
        color=color_col,
        color_discrete_map=color_discrete_map,
        title=title,
        opacity=0.8,
    )

    fig.update_layout(
        template="field_notebook",
        scene=dict(
            xaxis=dict(
                title="a_x (m/s²)", backgroundcolor=PALETTE["surface"], gridcolor=PALETTE["grid"]
            ),
            yaxis=dict(
                title="a_y (m/s²)", backgroundcolor=PALETTE["surface"], gridcolor=PALETTE["grid"]
            ),
            zaxis=dict(
                title="a_z (m/s²)", backgroundcolor=PALETTE["surface"], gridcolor=PALETTE["grid"]
            ),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
    )
    return fig


def plot_filter_frequency_response(
    cutoff_hz: float = BUTTERWORTH_CUTOFF_HZ,
    fs: float = ORIGINAL_SAMPLING_RATE_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> go.Figure:
    """Plot frequency response |H(f)| of the 4th-order zero-phase Butterworth filter."""
    nyquist = 0.5 * fs
    b, a = signal.butter(order, cutoff_hz / nyquist, btype="low")
    w, h = signal.freqz(b, a, worN=1024, fs=fs)

    # In zero-phase filtfilt, effective gain is |H(f)|^2
    gain_db = 20 * np.log10(np.maximum(1e-5, np.abs(h) ** 2))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=w,
            y=gain_db,
            mode="lines",
            name="Effective |H(f)|² (filtfilt)",
            line=dict(color=PALETTE["accent"], width=2.5),
        )
    )

    # Add -3dB and -6dB threshold guides
    fig.add_vline(
        x=cutoff_hz,
        line=dict(color="#8C5E58", dash="dot"),
        annotation_text=f"Cutoff {cutoff_hz} Hz",
    )
    fig.add_vline(
        x=12.5, line=dict(color="#B38D48", dash="dash"), annotation_text="Nyquist (25 Hz fs)"
    )

    apply_warm_layout(
        fig,
        title="Anti-Aliasing Butterworth Filter Frequency Response",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Attenuation (dB)",
        height=350,
    )
    fig.update_yaxes(range=[-60, 5])
    return fig


def plot_transition_matrix(
    A: np.ndarray,
    state_names: list[str] | None = None,
    title: str = "Transition Probability Matrix A",
) -> go.Figure:
    """Plot heatmap of state transition probabilities A_ij = P(S_j | S_i)."""
    names = state_names or list(STATE_NAMES)

    fig = go.Figure(
        data=go.Heatmap(
            z=A,
            x=[f"To: {s}" for s in names],
            y=[f"From: {s}" for s in names],
            colorscale=[
                [0.0, PALETTE["surface"]],
                [0.5, "#D9A441"],
                [1.0, PALETTE["accent"]],
            ],
            text=np.round(A, 4),
            texttemplate="%{text}",
            textfont={"size": 14, "color": PALETTE["text_primary"]},
            showscale=True,
        )
    )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Destination State (t+1)",
        yaxis_title="Source State (t)",
        height=380,
    )
    return fig


def plot_confusion_matrix_chart(
    cm_raw: list[list[int]] | np.ndarray,
    cm_norm: list[list[float]] | np.ndarray,
    state_names: list[str] | None = None,
    title: str = "Confusion Matrix (Held-out Test Participants)",
) -> go.Figure:
    """Plot normalized confusion matrix with annotations showing count and percentage."""
    names = state_names or list(STATE_NAMES)
    raw = np.asarray(cm_raw)
    norm = np.asarray(cm_norm)

    text_annotations = []
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            row.append(f"{raw[i, j]}<br>({norm[i, j] * 100:.1f}%)")
        text_annotations.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=norm,
            x=[f"Pred: {s}" for s in names],
            y=[f"True: {s}" for s in names],
            colorscale=[
                [0.0, PALETTE["surface"]],
                [0.5, "#D9A441"],
                [1.0, PALETTE["accent"]],
            ],
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 13, "color": PALETTE["text_primary"]},
            zmin=0.0,
            zmax=1.0,
        )
    )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Predicted Activity State",
        yaxis_title="Ground Truth Activity State",
        height=400,
    )
    return fig


def plot_reliability_diagram(
    calibration_data: dict,
    title: str = "Reliability Diagram (Posterior Confidence Calibration)",
) -> go.Figure:
    """Plot reliability diagram comparing confidence to empirical accuracy with ECE."""
    bins = calibration_data.get("bins", [])
    ece = calibration_data.get("ece", 0.0)

    confs = [b["confidence"] for b in bins if b["count"] > 0]
    accs = [b["accuracy"] for b in bins if b["count"] > 0]
    [b["count"] for b in bins if b["count"] > 0]

    fig = go.Figure()

    # Diagonal ideal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#B38D48", dash="dash", width=2),
        )
    )

    # Actual model calibration points
    fig.add_trace(
        go.Scatter(
            x=confs,
            y=accs,
            mode="markers+lines",
            name="Observed Accuracy",
            marker=dict(size=10, color=PALETTE["accent"], symbol="circle"),
            line=dict(color=PALETTE["accent"], width=2),
        )
    )

    apply_warm_layout(
        fig,
        title=f"{title} (ECE = {ece:.4f})",
        xaxis_title="Mean Predicted Posterior Confidence",
        yaxis_title="Empirical Accuracy",
        height=380,
    )
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    return fig


def plot_loso_comparison_chart(
    loso_data: dict,
    title: str = "Leave-One-Subject-Out (LOSO) Accuracy Across Participants",
) -> go.Figure:
    """Bar chart displaying cross-validation accuracy across each participant fold."""
    folds = loso_data.get("fold_results", [])
    subjects = [f"Subj {f['held_out_subject']}" for f in folds]
    accuracies = [f["accuracy"] * 100 for f in folds]
    mean_acc = loso_data.get("mean_accuracy", 0.0) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=accuracies,
            marker_color=PALETTE["neutral_data"],
            text=[f"{acc:.1f}%" for acc in accuracies],
            textposition="auto",
            name="Fold Accuracy",
        )
    )

    # Mean line
    fig.add_hline(
        y=mean_acc,
        line=dict(color=PALETTE["accent"], width=2, dash="dash"),
        annotation_text=f"Mean: {mean_acc:.1f}%",
    )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Held-out Test Subject",
        yaxis_title="Classification Accuracy (%)",
        height=360,
    )
    fig.update_yaxes(range=[0, 105])
    return fig
