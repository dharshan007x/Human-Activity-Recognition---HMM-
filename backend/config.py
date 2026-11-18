"""Configuration constants, design tokens, and paths for HAR HMM Backend."""

from pathlib import Path

# Base Directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_SAMPLE_PATH = DATA_DIR / "synthetic_demo_sample.csv"
DATASET_CARD_PATH = DATA_DIR / "dataset_card.md"

MODEL_ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
SAVED_HMM_PATH = MODEL_ARTIFACTS_DIR / "hmm_model_trained.json"
SAVED_SCALER_PATH = MODEL_ARTIFACTS_DIR / "scaler.json"
EVALUATION_RESULTS_PATH = MODEL_ARTIFACTS_DIR / "evaluation_results.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"

# Ensure runtime directories exist
for p in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_ARTIFACTS_DIR,
    REPORTS_DIR,
    SCREENSHOTS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# PAMAP2 Metadata and Dataset Specifications
PAMAP2_SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
)
PAMAP2_LICENCE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
ORIGINAL_SAMPLING_RATE_HZ = 100.0
TARGET_SAMPLING_RATE_HZ = 25.0
DECIMATION_FACTOR = int(ORIGINAL_SAMPLING_RATE_HZ / TARGET_SAMPLING_RATE_HZ)  # 4
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF_HZ = 10.0  # Safe anti-alias cutoff below 12.5 Hz Nyquist

# Selected IMU Placement: Chest IMU (+-16g accelerometer)
# Col 21, 22, 23: Chest IMU 3D acceleration +-16g (m/s^2)
CHEST_ACCEL_16G_COLS = [21, 22, 23]
CHEST_ACCEL_NAMES = ["acc_x", "acc_y", "acc_z"]
FEATURE_NAMES = ["acc_x", "acc_y", "acc_z", "acc_mag"]

# Physical bounds for validation
MAX_PHYSICAL_ACCEL_MS2 = 156.96  # +-16g limit

# Target Hidden States & Activity Mapping
TARGET_ACTIVITIES: dict[int, str] = {
    4: "Walking",
    2: "Sitting",
    5: "Running",
}

STATE_NAMES: list[str] = ["Walking", "Sitting", "Running"]
NUM_STATES: int = len(STATE_NAMES)
STATE_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(STATE_NAMES)}
INDEX_TO_STATE: dict[int, str] = {idx: name for idx, name in enumerate(STATE_NAMES)}

# Subject Splitting (Zero-Leakage Protocol)
ALL_SUBJECTS: list[int] = [101, 102, 103, 104, 105, 106, 107, 108, 109]
TRAIN_SUBJECTS: list[int] = [101, 102, 103, 104, 105, 106]
VAL_SUBJECTS: list[int] = [107]
TEST_SUBJECTS: list[int] = [108, 109]

# HMM Hyperparameters
LAPLACE_SMOOTHING: float = 1.0
COVARIANCE_REGULARIZATION: float = 1e-4
RANDOM_SEED: int = 42
SEED: int = 42

# Field Notebook Warm Design System Tokens
PALETTE = {
    "background": "#FBF6EE",  # Cream paper canvas
    "surface": "#F3EADB",  # Soft warm container/card
    "border": "#E2D3BC",  # Sand divider border
    "text_primary": "#2E2118",  # Deep umber
    "text_secondary": "#6B5A4A",  # Secondary text / captions
    "accent": "#C4603A",  # Terracotta action highlight
    "neutral_data": "#3F7F7A",  # Dusty teal for signal lines
    "grid": "#E2D3BC",  # Chart grid lines
    "success": "#4D7C5D",  # Warm muted green
    "warning": "#C78B2E",  # Warm muted amber
    "error": "#B84A39",  # Warm muted red
}

ACTIVITY_COLORS = {
    "Walking": "#6F8F72",  # Sage
    "Sitting": "#D9A441",  # Ochre
    "Running": "#C4603A",  # Terracotta
}

ACTIVITY_NAMES = TARGET_ACTIVITIES
STATE_ORDER = STATE_NAMES

