"""Service for running pure NumPy Forward, Backward, Posterior, and Viterbi inference."""

import time
import numpy as np

from backend.app.services.model_service import ModelService
from backend.config import STATE_NAMES


class InferenceService:
    def __init__(self):
        self.model_service = ModelService.get_instance()

    def _prepare_sequence(self, sequence: list[list[float]], scale_input: bool = False) -> np.ndarray:
        arr = np.array(sequence, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Input sequence must be 2-dimensional (T, D), got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("Input sequence cannot be empty.")
        if arr.shape[1] == 3:
            # Add magnitude feature if only 3 features provided
            mag = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.hstack([arr, mag])
        elif arr.shape[1] != 4:
            raise ValueError(f"Expected 3 or 4 accelerometer features, got {arr.shape[1]}")

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input sequence contains NaN or Infinite values.")

        if scale_input:
            scaler = self.model_service.scaler
            arr = scaler.transform([arr])[0]

        return arr

    def run_forward(self, sequence: list[list[float]], scale_input: bool = False) -> dict:
        t0 = time.perf_counter()
        X = self._prepare_sequence(sequence, scale_input=scale_input)
        hmm = self.model_service.hmm

        ll_log, log_alpha, _ = hmm.forward(X, method="log")
        ll_scaled, scaled_alpha, c_scale = hmm.forward(X, method="scaled")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "log_likelihood": float(ll_log),
            "log_alpha_table": log_alpha.tolist(),
            "scaled_alpha_table": scaled_alpha.tolist(),
            "scaling_factors": c_scale.tolist() if c_scale is not None else [],
            "state_names": STATE_NAMES,
            "time_ms": round(elapsed_ms, 3),
        }

    def run_backward(self, sequence: list[list[float]], scale_input: bool = False) -> dict:
        t0 = time.perf_counter()
        X = self._prepare_sequence(sequence, scale_input=scale_input)
        hmm = self.model_service.hmm

        ll_log, log_beta = hmm.backward(X, method="log")
        _, _, c_scale = hmm.forward(X, method="scaled")
        ll_scaled, scaled_beta = hmm.backward(X, method="scaled", c_scale=c_scale)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "log_likelihood": float(ll_log),
            "log_beta_table": log_beta.tolist(),
            "scaled_beta_table": scaled_beta.tolist(),
            "state_names": STATE_NAMES,
            "time_ms": round(elapsed_ms, 3),
        }

    def run_posterior(self, sequence: list[list[float]], scale_input: bool = False) -> dict:
        t0 = time.perf_counter()
        X = self._prepare_sequence(sequence, scale_input=scale_input)
        hmm = self.model_service.hmm

        gamma, pred_indices, ll, _, _ = hmm.posterior(X, method="log")
        predicted_states = [STATE_NAMES[idx] for idx in pred_indices]
        confidences = np.max(gamma, axis=1).tolist()
        row_sums = np.sum(gamma, axis=1)
        row_sums_ok = bool(np.allclose(row_sums, 1.0, atol=1e-5))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "log_likelihood": float(ll),
            "gamma_table": gamma.tolist(),
            "predicted_states": predicted_states,
            "confidences": [round(c, 4) for c in confidences],
            "row_sums_verified": row_sums_ok,
            "state_names": STATE_NAMES,
            "time_ms": round(elapsed_ms, 3),
        }

    def validate_likelihoods(self, sequence: list[list[float]], scale_input: bool = False) -> dict:
        X = self._prepare_sequence(sequence, scale_input=scale_input)
        hmm = self.model_service.hmm

        ll_fwd, _, _ = hmm.forward(X, method="log")
        ll_bwd, _ = hmm.backward(X, method="log")
        ll_scaled, _, _ = hmm.forward(X, method="scaled")

        diff_fwd_bwd = abs(ll_fwd - ll_bwd)
        diff_scaled = abs(ll_fwd - ll_scaled)
        passed = diff_fwd_bwd < 1e-8 and diff_scaled < 1e-6

        return {
            "forward_log_likelihood": round(float(ll_fwd), 8),
            "backward_log_likelihood": round(float(ll_bwd), 8),
            "scaled_forward_log_likelihood": round(float(ll_scaled), 8),
            "abs_difference_fwd_bwd": float(ll_fwd - ll_bwd),
            "abs_difference_log_scaled": float(diff_scaled),
            "tolerance_passed": passed,
            "status": "PASSED" if passed else "FAILED",
            "message": (
                "Forward and Backward algorithms produce mathematically identical sequence log-likelihoods "
                f"(discrepancy = {diff_fwd_bwd:.2e} < 1e-8 tolerance)."
                if passed
                else "Discrepancy exceeded acceptable numerical threshold."
            ),
        }

    def predict_full(self, sequence: list[list[float]], scale_input: bool = False) -> dict:
        t0 = time.perf_counter()
        X = self._prepare_sequence(sequence, scale_input=scale_input)
        hmm = self.model_service.hmm

        gamma, pointwise_indices, fwd_ll, _, _ = hmm.posterior(X, method="log")
        ll_bwd, _ = hmm.backward(X, method="log")
        vit_indices, vit_ll, _ = hmm.decode_viterbi(X)

        pointwise_states = [STATE_NAMES[i] for i in pointwise_indices]
        viterbi_states = [STATE_NAMES[i] for i in vit_indices]
        match_rate = float((pointwise_indices == vit_indices).mean() * 100.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pointwise_states": pointwise_states,
            "viterbi_states": viterbi_states,
            "viterbi_log_likelihood": float(vit_ll),
            "forward_log_likelihood": float(fwd_ll),
            "backward_log_likelihood": float(ll_bwd),
            "path_match_percentage": round(match_rate, 2),
            "gamma_table": gamma.tolist(),
            "confidences": [round(float(c), 4) for c in np.max(gamma, axis=1)],
            "state_names": STATE_NAMES,
            "time_ms": round(elapsed_ms, 3),
        }
