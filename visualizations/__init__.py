"""Visualization package with custom field_notebook warm theme."""

from .charts import (
    plot_3d_trajectory,
    plot_confusion_matrix_chart,
    plot_filter_frequency_response,
    plot_loso_comparison_chart,
    plot_reliability_diagram,
    plot_transition_matrix,
    plot_triaxial_signals,
)
from .theme import apply_warm_layout, register_field_notebook_theme
from .trellis import (
    plot_backward_trellis,
    plot_forward_trellis,
    plot_posterior_trellis,
    plot_step_probabilities_bars,
)

__all__ = [
    "register_field_notebook_theme",
    "apply_warm_layout",
    "plot_triaxial_signals",
    "plot_3d_trajectory",
    "plot_filter_frequency_response",
    "plot_transition_matrix",
    "plot_confusion_matrix_chart",
    "plot_reliability_diagram",
    "plot_loso_comparison_chart",
    "plot_forward_trellis",
    "plot_backward_trellis",
    "plot_posterior_trellis",
    "plot_step_probabilities_bars",
]

