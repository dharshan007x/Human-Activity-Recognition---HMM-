"""Custom 'field_notebook' warm Plotly theme and chart helpers."""

import plotly.graph_objects as go
import plotly.io as pio

from config import PALETTE


def register_field_notebook_theme() -> str:
    """Create and register the warm 'field_notebook' Plotly template.

    Evokes a high-quality science monograph or field notebook:
    - Background: warm cream paper (#F3EADB)
    - Grid lines: subtle sand (#E2D3BC)
    - Font: Fraunces for titles, Source Sans 3 for labels and ticks
    - Palette: sage (walking), ochre (sitting), terracotta (running), dusty teal (signal)
    """
    theme_name = "field_notebook"

    template = go.layout.Template()

    template.layout = go.Layout(
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(
            family="Source Sans 3, Georgia, serif",
            size=13,
            color=PALETTE["text_primary"],
        ),
        title=dict(
            font=dict(
                family="Fraunces, Georgia, serif",
                size=18,
                color=PALETTE["text_primary"],
            ),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["text_secondary"], size=11),
            title=dict(
                font=dict(
                    family="Source Sans 3, sans-serif",
                    color=PALETTE["text_primary"],
                    size=13,
                )
            ),
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["text_secondary"], size=11),
            title=dict(
                font=dict(
                    family="Source Sans 3, sans-serif",
                    color=PALETTE["text_primary"],
                    size=13,
                )
            ),
        ),
        legend=dict(
            bgcolor="rgba(243, 234, 219, 0.8)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(color=PALETTE["text_primary"], size=12),
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=PALETTE["surface"],
            bordercolor=PALETTE["border"],
            font=dict(family="Source Sans 3, sans-serif", color=PALETTE["text_primary"]),
        ),
    )

    pio.templates[theme_name] = template
    pio.templates.default = theme_name
    return theme_name


# Automatically register on module import
register_field_notebook_theme()


def apply_warm_layout(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 400,
) -> go.Figure:
    """Convenience helper to apply consistent warm layout styling to any figure."""
    fig.update_layout(
        template="field_notebook",
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    return fig
