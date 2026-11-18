"""Pydantic v2 schemas for HMM inference endpoints."""

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="Observation feature sequence of shape (T, 4) with [acc_x, acc_y, acc_z, acc_mag]",
    )
    scale_input: bool = Field(
        default=False,
        description="Whether to apply fitted training scaler prior to inference",
    )


class ForwardResponse(BaseModel):
    log_likelihood: float
    log_alpha_table: list[list[float]]
    scaled_alpha_table: list[list[float]]
    scaling_factors: list[float]
    state_names: list[str]
    time_ms: float


class BackwardResponse(BaseModel):
    log_likelihood: float
    log_beta_table: list[list[float]]
    scaled_beta_table: list[list[float]]
    state_names: list[str]
    time_ms: float


class PosteriorResponse(BaseModel):
    log_likelihood: float
    gamma_table: list[list[float]]
    predicted_states: list[str]
    confidences: list[float]
    row_sums_verified: bool
    state_names: list[str]
    time_ms: float


class ValidationResponse(BaseModel):
    forward_log_likelihood: float
    backward_log_likelihood: float
    scaled_forward_log_likelihood: float
    abs_difference_fwd_bwd: float
    abs_difference_log_scaled: float
    tolerance_passed: bool
    status: str
    message: str


class PredictResponse(BaseModel):
    pointwise_states: list[str]
    viterbi_states: list[str]
    viterbi_log_likelihood: float
    forward_log_likelihood: float
    backward_log_likelihood: float
    path_match_percentage: float
    gamma_table: list[list[float]]
    confidences: list[float]
    state_names: list[str]
    time_ms: float


class SampleSequenceItem(BaseModel):
    id: str
    subject_id: int
    activity_name: str
    length: int
    raw_sequence: list[list[float]]
    scaled_sequence: list[list[float]]
