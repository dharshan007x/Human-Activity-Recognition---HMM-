"""Subject-wise data splitting, scaler isolation, and Leave-One-Subject-Out cross-validation."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from config import (
    SAVED_SCALER_PATH,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
)


class SubjectScaler:
    """Standard scaler fit strictly on training subjects to prevent data leakage."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def fit(
        self, sequences: list[np.ndarray], feature_names: list[str] | None = None
    ) -> "SubjectScaler":
        """Compute mean and standard deviation across all time steps in training sequences."""
        if not sequences:
            raise ValueError("Cannot fit scaler on empty sequence list.")

        concatenated = np.vstack(sequences)  # (Total_T, D)
        self.mean_ = np.mean(concatenated, axis=0)
        self.scale_ = np.std(concatenated, axis=0)
        # Prevent division by zero on static channels
        self.scale_[self.scale_ < 1e-8] = 1.0

        if feature_names:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(len(self.mean_))]
        return self

    def transform(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        """Apply standardization z = (x - mu) / sigma to each sequence."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SubjectScaler must be fit before calling transform.")

        transformed = []
        for seq in sequences:
            z = (seq - self.mean_) / self.scale_
            transformed.append(z)
        return transformed

    def fit_transform(
        self, sequences: list[np.ndarray], feature_names: list[str] | None = None
    ) -> list[np.ndarray]:
        return self.fit(sequences, feature_names=feature_names).transform(sequences)

    def save(self, file_path: Path | str = SAVED_SCALER_PATH) -> None:
        """Serialize scaler parameters to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
            "feature_names": self.feature_names_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: Path | str = SAVED_SCALER_PATH) -> "SubjectScaler":
        """Deserialize scaler parameters from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler parameter file not found: {path}")

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        scaler = cls()
        scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.array(payload["scale"], dtype=np.float64)
        scaler.feature_names_ = payload.get("feature_names", [])
        return scaler


class SubjectSplitter:
    """Manages participant-wise splits and Leave-One-Subject-Out cross-validation."""

    def __init__(
        self,
        train_subjects: list[int] | None = None,
        val_subjects: list[int] | None = None,
        test_subjects: list[int] | None = None,
    ) -> None:
        self.train_subjects = train_subjects or TRAIN_SUBJECTS
        self.val_subjects = val_subjects or VAL_SUBJECTS
        self.test_subjects = test_subjects or TEST_SUBJECTS

    def split(
        self,
        sequences: list[np.ndarray],
        labels: list[int],
        subject_ids: list[int],
    ) -> tuple[
        tuple[list[np.ndarray], list[int]],
        tuple[list[np.ndarray], list[int]],
        tuple[list[np.ndarray], list[int]],
    ]:
        """Split sequences into train, validation, and test subsets based on subject IDs."""
        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []

        for seq, label, sid in zip(sequences, labels, subject_ids):
            if sid in self.train_subjects:
                train_x.append(seq)
                train_y.append(label)
            elif sid in self.val_subjects:
                val_x.append(seq)
                val_y.append(label)
            elif sid in self.test_subjects:
                test_x.append(seq)
                test_y.append(label)

        # Fallback if dataset only contains a subset of subjects (e.g. synthetic sample)
        if not train_x and sequences:
            # Partition existing subjects 70% train, 30% test
            unique_sids = sorted(list(set(subject_ids)))
            mid = max(1, int(len(unique_sids) * 0.7))
            auto_train = set(unique_sids[:mid])
            auto_test = set(unique_sids[mid:])

            for seq, label, sid in zip(sequences, labels, subject_ids):
                if sid in auto_train:
                    train_x.append(seq)
                    train_y.append(label)
                elif sid in auto_test:
                    test_x.append(seq)
                    test_y.append(label)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def loso_folds(
        self,
        sequences: list[np.ndarray],
        labels: list[int],
        subject_ids: list[int],
    ) -> Iterator[
        tuple[int, tuple[list[np.ndarray], list[int]], tuple[list[np.ndarray], list[int]]]
    ]:
        """Generate Leave-One-Subject-Out (LOSO) cross-validation folds.

        Yields:
            (held_out_subject_id, (train_x, train_y), (test_x, test_y))
        """
        unique_subjects = sorted(list(set(subject_ids)))
        for held_out in unique_subjects:
            train_x, train_y = [], []
            test_x, test_y = [], []

            for seq, label, sid in zip(sequences, labels, subject_ids):
                if sid == held_out:
                    test_x.append(seq)
                    test_y.append(label)
                else:
                    train_x.append(seq)
                    train_y.append(label)

            if train_x and test_x:
                yield held_out, (train_x, train_y), (test_x, test_y)
