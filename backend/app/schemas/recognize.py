"""Pydantic v2 schemas for CSV recognition endpoint."""

from pydantic import BaseModel


class StepPrediction(BaseModel):
    step: int
    t_sec: float
    acc_x: float
    acc_y: float
    acc_z: float
    acc_mag: float
    predicted_activity: str
    confidence: float
    walking_prob: float
    sitting_prob: float
    running_prob: float


class RecognitionResponse(BaseModel):
    filename: str
    total_timesteps: int
    duration_seconds: float
    overall_predicted_activity: str
    activity_distribution: dict[str, float]
    mean_confidence: float
    forward_log_likelihood: float
    viterbi_log_likelihood: float
    steps: list[StepPrediction]
    status: str
    message: str
