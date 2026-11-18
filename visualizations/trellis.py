"""Trellis diagrams and dynamic time-step probability visualization."""

import numpy as np
import plotly.graph_objects as go

from config import ACTIVITY_COLORS, PALETTE, STATE_NAMES
from visualizations.theme import apply_warm_layout


def plot_forward_trellis(
    log_alpha: np.ndarray,
    state_names: list[str] | None = None,
    title: str = "Forward Variable Trellis log α_t(i)",
) -> go.Figure:
    """Plot the forward log-variable log alpha_t(i) across all time steps t."""
    names = state_names or list(STATE_NAMES)
    T, N = log_alpha.shape
    time_steps = np.arange(1, T + 1)

    fig = go.Figure()

    for i in range(N):
        name = names[i]
        color = ACTIVITY_COLORS.get(name, PALETTE["accent"])
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=log_alpha[:, i],
                mode="lines+markers",
                marker=dict(size=4),
                name=f"log α({name})",
                line=dict(color=color, width=2.0),
            )
        )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Time Step t (Observation Sequence)",
        yaxis_title="Log Forward Probability log α_t(i)",
        height=380,
    )
    return fig


def plot_backward_trellis(
    log_beta: np.ndarray,
    state_names: list[str] | None = None,
    title: str = "Backward Variable Trellis log β_t(i)",
) -> go.Figure:
    """Plot backward log-variable log beta_t(i) across all time steps t."""
    names = state_names or list(STATE_NAMES)
    T, N = log_beta.shape
    time_steps = np.arange(1, T + 1)

    fig = go.Figure()

    for i in range(N):
        name = names[i]
        color = ACTIVITY_COLORS.get(name, PALETTE["accent"])
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=log_beta[:, i],
                mode="lines+markers",
                marker=dict(size=4),
                name=f"log β({name})",
                line=dict(color=color, width=2.0),
            )
        )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Time Step t (Observation Sequence)",
        yaxis_title="Log Backward Probability log β_t(i)",
        height=380,
    )
    return fig


def plot_posterior_trellis(
    gamma: np.ndarray,
    state_names: list[str] | None = None,
    title: str = "Posterior State Probability Smoothing γ_t(i)",
) -> go.Figure:
    """Plot state posteriors gamma_t(i) = P(q_t = S_i | O, lambda) over time."""
    names = state_names or list(STATE_NAMES)
    T, N = gamma.shape
    time_steps = np.arange(1, T + 1)

    fig = go.Figure()

    for i in range(N):
        name = names[i]
        color = ACTIVITY_COLORS.get(name, PALETTE["accent"])
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=gamma[:, i],
                mode="lines",
                name=f"P({name} | O)",
                line=dict(color=color, width=2.5),
            )
        )

    apply_warm_layout(
        fig,
        title=title,
        xaxis_title="Time Step t (Observation Sequence)",
        yaxis_title="Posterior State Probability γ_t(i)",
        height=380,
    )
    fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def plot_step_probabilities_bars(
    probs: np.ndarray,
    step_index: int,
    var_name: str = "γ",
    state_names: list[str] | None = None,
) -> go.Figure:
    """Bar chart showing probabilities across all hidden states at a chosen time step."""
    names = state_names or list(STATE_NAMES)
    colors = [ACTIVITY_COLORS.get(n, PALETTE["accent"]) for n in names]

    fig = go.Figure(
        data=go.Bar(
            x=names,
            y=probs,
            marker_color=colors,
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="auto",
        )
    )

    apply_warm_layout(
        fig,
        title=f"State Distribution at Time Step t = {step_index} ({var_name}_t)",
        xaxis_title="Hidden Activity State",
        yaxis_title=f"Probability {var_name}_t(i)",
        height=320,
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig
