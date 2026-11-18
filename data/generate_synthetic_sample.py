"""Generate a physically realistic synthetic demo dataset matching PAMAP2 chest IMU dynamics.

Clearly labeled as SYNTHETIC DEMO SAMPLE for offline testing and Streamlit demonstrations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import RANDOM_SEED, SYNTHETIC_SAMPLE_PATH, TARGET_SAMPLING_RATE_HZ


def generate_synthetic_data(
    duration_per_activity_sec: float = 60.0,
    fs: float = TARGET_SAMPLING_RATE_HZ,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate sequential triaxial acceleration readings for Sitting, Walking, and Running.

    Physics model:
    - Sitting (ID=2): Static gravity vector pointing along Y axis (body vertical) with
      subtle respiratory noise (variance ~ 0.04 m/s^2).
    - Walking (ID=4): 1.8 Hz cadence, harmonic gait cycle, dynamic range 8 to 15 m/s^2.
    - Running (ID=5): 2.8 Hz cadence, high ballistic ground-reaction spikes up to 28 m/s^2.
    """
    rng = np.random.default_rng(seed)
    n_samples_per_act = int(duration_per_activity_sec * fs)
    records = []
    current_time = 0.0
    dt = 1.0 / fs

    activities = []
    # Generate across all 9 PAMAP2 subjects (101 to 109)
    for sid in range(101, 110):
        activities.append((2, "Sitting", sid))
        activities.append((4, "Walking", sid))
        activities.append((5, "Running", sid))

    for act_id, act_name, subject_id in activities:
        t = np.linspace(0, duration_per_activity_sec, n_samples_per_act, endpoint=False)

        if act_name == "Sitting":
            # Gravity primarily in +Y (torso upright ~ 9.6 m/s^2), slight tilt on X and Z
            ax = 0.40 + 0.08 * rng.standard_normal(n_samples_per_act)
            ay = 9.65 + 0.12 * rng.standard_normal(n_samples_per_act)
            az = 1.10 + 0.08 * rng.standard_normal(n_samples_per_act)

        elif act_name == "Walking":
            # Cadence: 1.8 Hz step frequency + 3.6 Hz harmonic
            cadence = 1.8
            phase = rng.uniform(0, 2 * np.pi)
            ay_dyn = 3.5 * np.sin(2 * np.pi * cadence * t + phase) + 1.2 * np.sin(
                4 * np.pi * cadence * t
            )
            ax_dyn = 1.6 * np.sin(2 * np.pi * cadence * t + phase + np.pi / 4)
            az_dyn = 1.2 * np.cos(2 * np.pi * cadence * t + phase)
            noise = 0.35 * rng.standard_normal(n_samples_per_act)

            ax = 0.50 + ax_dyn + noise
            ay = 9.81 + ay_dyn + noise
            az = 1.20 + az_dyn + noise

        elif act_name == "Running":
            # Cadence: 2.8 Hz step frequency with asymmetric heel-strike impact spikes
            cadence = 2.8
            phase = rng.uniform(0, 2 * np.pi)
            strike = np.maximum(0, np.sin(2 * np.pi * cadence * t + phase)) ** 3
            ay_dyn = 11.5 * strike - 4.2 * np.cos(2 * np.pi * cadence * t + phase)
            ax_dyn = 3.8 * np.sin(2 * np.pi * cadence * t + phase)
            az_dyn = 2.5 * np.cos(2 * np.pi * cadence * t + phase)
            noise = 0.85 * rng.standard_normal(n_samples_per_act)

            ax = 0.80 + ax_dyn + noise
            ay = 9.81 + ay_dyn + noise
            az = 1.50 + az_dyn + noise

        mag = np.sqrt(ax**2 + ay**2 + az**2)

        for i in range(n_samples_per_act):
            records.append(
                {
                    "timestamp": round(current_time, 4),
                    "subject_id": subject_id,
                    "activity_id": act_id,
                    "activity_name": act_name,
                    "acc_x": round(float(ax[i]), 4),
                    "acc_y": round(float(ay[i]), 4),
                    "acc_z": round(float(az[i]), 4),
                    "acc_mag": round(float(mag[i]), 4),
                }
            )
            current_time += dt

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    SYNTHETIC_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_SAMPLE_PATH, index=False)
    print(f"[SUCCESS] Generated synthetic demo dataset at {SYNTHETIC_SAMPLE_PATH}")
    print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")
    print(df.groupby(["subject_id", "activity_name"]).size())
