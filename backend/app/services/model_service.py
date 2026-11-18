"""Service for managing trained HMM model and scaler singletons."""

from backend.core.hmm_model import GaussianHMM
from backend.preprocessing.split import SubjectScaler
from backend.config import SAVED_HMM_PATH, SAVED_SCALER_PATH, STATE_NAMES, FEATURE_NAMES


class ModelService:
    _instance = None
    _hmm: GaussianHMM | None = None
    _scaler: SubjectScaler | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self):
        if SAVED_HMM_PATH.exists():
            self._hmm = GaussianHMM.load(SAVED_HMM_PATH)
        else:
            self._hmm = None

        if SAVED_SCALER_PATH.exists():
            self._scaler = SubjectScaler.load(SAVED_SCALER_PATH)
        else:
            self._scaler = None

    @property
    def hmm(self) -> GaussianHMM:
        if self._hmm is None:
            self.load_artifacts()
            if self._hmm is None:
                raise RuntimeError("HMM model artifact not found. Please train model first.")
        return self._hmm

    @property
    def scaler(self) -> SubjectScaler:
        if self._scaler is None:
            self.load_artifacts()
            if self._scaler is None:
                raise RuntimeError("Scaler artifact not found. Please fit scaler first.")
        return self._scaler

    def get_parameters_dict(self) -> dict:
        hmm = self.hmm
        return {
            "n_states": hmm.n_states,
            "state_names": STATE_NAMES,
            "feature_names": FEATURE_NAMES,
            "pi": hmm.pi.tolist(),
            "A": hmm.A.tolist(),
            "means": hmm.means.tolist(),
            "covars": hmm.covars.tolist(),
            "reg_covar": hmm.reg_cov,
            "is_fitted": hmm.is_fitted,
        }
