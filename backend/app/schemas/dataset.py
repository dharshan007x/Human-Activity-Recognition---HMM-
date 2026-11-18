"""Pydantic v2 schemas for dataset and signal endpoints."""

from pydantic import BaseModel, Field


class DatasetMetadataResponse(BaseModel):
    name: str = "PAMAP2 Physical Activity Monitoring Benchmark"
    source_url: str
    licence: str
    subjects: list[int]
    train_subjects: list[int]
    val_subjects: list[int]
    test_subjects: list[int]
    original_sampling_rate_hz: float
    target_sampling_rate_hz: float
    decimation_factor: int
    filter_type: str = "4th-order zero-phase Butterworth low-pass"
    filter_cutoff_hz: float
    chosen_sensor: str = "Chest IMU ±16g Accelerometer (Cols 21-23)"
    chosen_sensor_justification: str
    target_activities: dict[int, str]
    feature_names: list[str]


class DatasetSummaryResponse(BaseModel):
    total_records: int
    total_processed_samples: int | None = None
    sequence_count: int | None = None
    total_subjects: int
    activities_present: list[str]
    records_per_activity: dict[str, int]
    activity_distribution: dict[str, int] | None = None
    missing_values_count: int
    clean_status: str


class DataPreviewRow(BaseModel):
    timestamp_s: float
    timestamp: float | None = None
    activity_id: int
    activity_name: str
    subject_id: int
    acc_x: float
    acc_y: float
    acc_z: float
    acc_mag: float


class DatasetPreviewResponse(BaseModel):
    total_rows: int
    total_records: int | None = None
    limit: int
    offset: int
    rows: list[DataPreviewRow]
    records: list[DataPreviewRow] | None = None
    columns: list[str] | None = None


class SignalPoint(BaseModel):
    t_sec: float
    acc_x: float
    acc_y: float
    acc_z: float
    acc_mag: float


class SignalStreamResponse(BaseModel):
    subject_id: int
    activity_name: str
    sampling_rate_hz: float
    points: list[SignalPoint]
