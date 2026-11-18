"""Application configuration for FastAPI backend."""

from pathlib import Path
from backend.config import (
    BACKEND_DIR,
    PROJECT_ROOT,
    DATA_DIR,
    MODEL_ARTIFACTS_DIR,
    SAVED_HMM_PATH,
    SAVED_SCALER_PATH,
    EVALUATION_RESULTS_PATH,
    SYNTHETIC_SAMPLE_PATH,
    DATASET_CARD_PATH,
    TARGET_ACTIVITIES,
    STATE_NAMES,
    STATE_ORDER,
    CHEST_ACCEL_NAMES,
    FEATURE_NAMES,
    DECIMATION_FACTOR,
    TARGET_SAMPLING_RATE_HZ,
    ORIGINAL_SAMPLING_RATE_HZ,
    BUTTERWORTH_CUTOFF_HZ,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    ALL_SUBJECTS,
    PALETTE,
    ACTIVITY_COLORS,
)

API_PREFIX = "/api/v1"
TITLE = "Human Activity Recognition API"
VERSION = "1.0.0"
DESCRIPTION = "Production scientific API for Human Activity Recognition using Forward & Backward Algorithms."
