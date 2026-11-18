"""Service for dataset metadata, preview, signal streams, and sample sequences."""

import numpy as np
import pandas as pd
from scipy.signal import butter, freqz

from backend.config import (
    PAMAP2_SOURCE_URL,
    PAMAP2_LICENCE,
    ALL_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    ORIGINAL_SAMPLING_RATE_HZ,
    TARGET_SAMPLING_RATE_HZ,
    DECIMATION_FACTOR,
    BUTTERWORTH_CUTOFF_HZ,
    TARGET_ACTIVITIES,
    FEATURE_NAMES,
    STATE_NAMES,
)
from backend.preprocessing.loader import load_data
from backend.app.services.model_service import ModelService


class DatasetService:
    _instance = None
    _processed_df: pd.DataFrame | None = None
    _sample_sequences: list[dict] | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_processed_data(self) -> pd.DataFrame:
        if self._processed_df is None:
            self._processed_df = load_data()
        return self._processed_df

    def get_metadata(self) -> dict:
        return {
            "name": "PAMAP2 Physical Activity Monitoring Benchmark",
            "source_url": PAMAP2_SOURCE_URL,
            "licence": PAMAP2_LICENCE,
            "subjects": ALL_SUBJECTS,
            "train_subjects": TRAIN_SUBJECTS,
            "val_subjects": VAL_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
            "original_sampling_rate_hz": ORIGINAL_SAMPLING_RATE_HZ,
            "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
            "decimation_factor": DECIMATION_FACTOR,
            "filter_type": "4th-order zero-phase Butterworth low-pass",
            "filter_cutoff_hz": BUTTERWORTH_CUTOFF_HZ,
            "chosen_sensor": "Chest IMU ±16g Accelerometer (Cols 21-23)",
            "chosen_sensor_justification": (
                "The chest sits on the thoracic center of mass, isolating whole-body posture and locomotion "
                "from voluntary peripheral arm gestures (hand IMU) and ground impact spikes (ankle IMU). "
                "The ±16g dynamic range prevents impact saturation during running heel-strikes."
            ),
            "target_activities": TARGET_ACTIVITIES,
            "feature_names": FEATURE_NAMES,
        }

    def get_summary(self) -> dict:
        df = self.get_processed_data()
        counts = df["activity_name"].value_counts().to_dict()
        return {
            "total_records": len(df),
            "total_processed_samples": len(df),
            "sequence_count": len(df) // 50,
            "total_subjects": int(df["subject_id"].nunique()),
            "activities_present": sorted(df["activity_name"].unique().tolist()),
            "records_per_activity": counts,
            "activity_distribution": counts,
            "missing_values_count": int(df.isna().sum().sum()),
            "clean_status": "Cleaned (Zero NaNs, Bounds Checked ±16g)",
        }

    def get_preview(self, limit: int = 50, offset: int = 0) -> dict:
        df = self.get_processed_data()
        subset = df.iloc[offset : offset + limit]
        rows = []
        for _, row in subset.iterrows():
            ts = float(row.get("timestamp", row.get("timestamp_s", 0.0)))
            rows.append(
                {
                    "timestamp_s": ts,
                    "timestamp": ts,
                    "activity_id": int(row["activity_id"]),
                    "activity_name": str(row["activity_name"]),
                    "subject_id": int(row["subject_id"]),
                    "acc_x": float(row["acc_x"]),
                    "acc_y": float(row["acc_y"]),
                    "acc_z": float(row["acc_z"]),
                    "acc_mag": float(row["acc_mag"]),
                }
            )
        return {
            "total_rows": len(df),
            "total_records": len(df),
            "limit": limit,
            "offset": offset,
            "rows": rows,
            "records": rows,
            "columns": ["subject_id", "activity_name", "timestamp_s", "acc_x", "acc_y", "acc_z", "acc_mag"],
        }

    def get_signals(self, subject_id: int = 101, activity_name: str = "Walking", max_points: int = 150) -> dict:
        df = self.get_processed_data()
        sub = df[(df["subject_id"] == subject_id) & (df["activity_name"] == activity_name)]
        if sub.empty:
            sub = df[df["activity_name"] == activity_name]
        points = []
        subset = sub.iloc[:max_points]
        for idx, (_, row) in enumerate(subset.iterrows()):
            points.append(
                {
                    "t_sec": round(idx / TARGET_SAMPLING_RATE_HZ, 3),
                    "acc_x": round(float(row["acc_x"]), 4),
                    "acc_y": round(float(row["acc_y"]), 4),
                    "acc_z": round(float(row["acc_z"]), 4),
                    "acc_mag": round(float(row["acc_mag"]), 4),
                }
            )
        return {
            "subject_id": subject_id,
            "activity_name": activity_name,
            "sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
            "points": points,
        }

    def get_butterworth_response(self) -> dict:
        b, a = butter(4, BUTTERWORTH_CUTOFF_HZ / (ORIGINAL_SAMPLING_RATE_HZ / 2.0), btype="low")
        w, h = freqz(b, a, worN=256, fs=ORIGINAL_SAMPLING_RATE_HZ)
        freqs = w.tolist()
        gains_db = (20 * np.log10(np.maximum(abs(h), 1e-5))).tolist()
        return {
            "order": 4,
            "cutoff_hz": BUTTERWORTH_CUTOFF_HZ,
            "original_fs": ORIGINAL_SAMPLING_RATE_HZ,
            "frequencies": [round(f, 2) for f in freqs],
            "gains_db": [round(g, 2) for g in gains_db],
            "magnitude_db": [round(g, 2) for g in gains_db],
        }

    def get_sample_sequences(self) -> list[dict]:
        if self._sample_sequences is not None:
            return self._sample_sequences

        df = self.get_processed_data()
        model_service = ModelService.get_instance()
        scaler = model_service.scaler

        samples = []
        for act in STATE_NAMES:
            test_sub = df[(df["subject_id"].isin(TEST_SUBJECTS)) & (df["activity_name"] == act)]
            if len(test_sub) < 50:
                test_sub = df[df["activity_name"] == act]
            seq_raw = test_sub[["acc_x", "acc_y", "acc_z", "acc_mag"]].iloc[:50].values
            seq_scaled = scaler.transform([seq_raw])[0]
            samples.append(
                {
                    "id": f"sample_{act.lower()}",
                    "subject_id": int(test_sub["subject_id"].iloc[0]),
                    "activity_name": act,
                    "length": len(seq_raw),
                    "raw_sequence": seq_raw.tolist(),
                    "scaled_sequence": seq_scaled.tolist(),
                }
            )
        self._sample_sequences = samples
        return self._sample_sequences
