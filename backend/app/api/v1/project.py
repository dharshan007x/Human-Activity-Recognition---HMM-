"""Project overview and summary endpoints."""

from fastapi import APIRouter
from backend.config import STATE_NAMES, FEATURE_NAMES, PALETTE, ACTIVITY_COLORS

router = APIRouter(prefix="/project", tags=["Project"])


@router.get("/summary")
def get_project_summary():
    return {
        "title": "Human Activity Recognition Using the Forward and Backward Algorithms",
        "official_problem_statement": (
            "Identify human activities such as walking, sitting and running from sequential "
            "accelerometer readings. Model activities as hidden states and sensor patterns as "
            "observations, then use the Forward and Backward Algorithms to calculate sequence "
            "likelihoods and state probabilities."
        ),
        "hidden_states": STATE_NAMES,
        "observation_features": FEATURE_NAMES,
        "model_architecture": "First-Order Gaussian Hidden Markov Model λ = (π, A, B)",
        "inference_engine": "Pure NumPy Manual Implementations (Forward, Backward, Posterior, Viterbi)",
        "design_theme": {
            "palette": PALETTE,
            "activity_colors": ACTIVITY_COLORS,
        },
    }
