"""Pydantic v2 schemas for model evaluation and benchmark metrics."""

from pydantic import BaseModel


class PerClassMetric(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class ConfusionMatrixData(BaseModel):
    classes: list[str]
    raw: list[list[int]]
    normalized: list[list[float]]


class CalibrationBin(BaseModel):
    bin_index: int
    confidence: float
    accuracy: float
    count: int


class BootstrapCI(BaseModel):
    metric: str
    mean: float
    lower_95: float
    upper_95: float


class LOSOFold(BaseModel):
    fold: int
    test_subject: int
    accuracy: float
    macro_f1: float


class EvaluationSummaryResponse(BaseModel):
    test_participants: list[int]
    total_test_sequences: int
    total_time_steps: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    expected_calibration_error: float
    mean_sequence_log_likelihood: float
    log_likelihood_std: float
    per_class: dict[str, PerClassMetric]
    confusion_matrix: ConfusionMatrixData
    calibration_bins: list[CalibrationBin]
    bootstrap_cis: list[BootstrapCI]
    loso_folds: list[LOSOFold]
    loso_mean_accuracy: float
    loso_std_accuracy: float
