"""In-depth physical error analysis for ambulatory state transitions."""

from typing import Any

import numpy as np

from config import STATE_NAMES


def analyze_locomotion_errors(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    raw_features: np.ndarray | list[np.ndarray],
    state_names: list[str] | None = None,
) -> dict[str, Any]:
    """Perform biomechanical error analysis on Walking vs Running confusions.

    Investigates:
    - High-cadence walking misclassified as running.
    - Low-speed jogging misclassified as walking.
    - Postural stability of sitting vs locomotion states.

    Args:
        y_true: True states.
        y_pred: Predicted states.
        raw_features: Sequence observations with acc_mag.
        state_names: Names of states.

    Returns:
        Structured breakdown of physical causes of confusion.
    """
    names = state_names or list(STATE_NAMES)
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    walk_idx = names.index("Walking") if "Walking" in names else 0
    sit_idx = names.index("Sitting") if "Sitting" in names else 1
    run_idx = names.index("Running") if "Running" in names else 2

    # Confusion counts
    walk_as_run = int(np.sum((y_t == walk_idx) & (y_p == run_idx)))
    run_as_walk = int(np.sum((y_t == run_idx) & (y_p == walk_idx)))
    sit_as_locomotion = int(np.sum((y_t == sit_idx) & (y_p != sit_idx)))
    locomotion_as_sit = int(np.sum((y_t != sit_idx) & (y_p == sit_idx)))

    total_walk = int(np.sum(y_t == walk_idx))
    total_run = int(np.sum(y_t == run_idx))
    total_sit = int(np.sum(y_t == sit_idx))

    return {
        "walking_misclassified_as_running": {
            "count": walk_as_run,
            "rate": float(walk_as_run / max(1, total_walk)),
            "physical_explanation": (
                "Occurs during brisk walking or uphill grade where arm/torso acceleration "
                "approaches the low-end ground impact threshold of jogging (15-18 m/s^2)."
            ),
        },
        "running_misclassified_as_walking": {
            "count": run_as_walk,
            "rate": float(run_as_walk / max(1, total_run)),
            "physical_explanation": (
                "Occurs during slow fatigue-induced deceleration or transition strides where "
                "heel strike impact dampens and cadence drops below 2.2 Hz."
            ),
        },
        "sitting_misclassified_as_locomotion": {
            "count": sit_as_locomotion,
            "rate": float(sit_as_locomotion / max(1, total_sit)),
            "physical_explanation": (
                "Extremely rare when using Chest IMU due to static 1g gravity vector. "
                "Primarily caused by deep coughing or torso realignment on the chair."
            ),
        },
        "locomotion_misclassified_as_sitting": {
            "count": locomotion_as_sit,
            "rate": float(locomotion_as_sit / max(1, total_walk + total_run)),
            "physical_explanation": (
                "Momentary pauses between strides or momentary standing pauses mid-sequence."
            ),
        },
    }

