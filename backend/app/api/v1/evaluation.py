"""Model evaluation and benchmark metrics API endpoint."""

import json
from fastapi import APIRouter, HTTPException

from backend.config import EVALUATION_RESULTS_PATH, TEST_SUBJECTS
from backend.app.schemas.evaluation import (
    EvaluationSummaryResponse,
    PerClassMetric,
    ConfusionMatrixData,
    CalibrationBin,
    BootstrapCI,
    LOSOFold,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/summary", response_model=EvaluationSummaryResponse)
def get_evaluation_summary():
    if not EVALUATION_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results artifact not found. Please run model evaluation first.",
        )

    with open(EVALUATION_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    test_metrics = data["test_metrics"]
    per_class_dict = {}
    for cls_name, m in test_metrics["per_class"].items():
        per_class_dict[cls_name] = PerClassMetric(
            precision=round(float(m["precision"]), 4),
            recall=round(float(m["recall"]), 4),
            f1_score=round(float(m["f1_score"]), 4),
            support=int(m["support"]),
        )

    cm_data = ConfusionMatrixData(
        classes=test_metrics["state_names"],
        raw=test_metrics["confusion_matrix_raw"],
        normalized=[[round(float(v), 4) for v in row] for row in test_metrics["confusion_matrix_normalized"]],
    )

    calib_bins = []
    for b in data.get("calibration", {}).get("bins", []):
        calib_bins.append(
            CalibrationBin(
                bin_index=int(b["bin_index"]),
                confidence=round(float(b["confidence"]), 4),
                accuracy=round(float(b["accuracy"]), 4),
                count=int(b["count"]),
            )
        )

    bootstrap_cis = []
    bs = data.get("bootstrap", {})
    if "accuracy" in bs:
        bootstrap_cis.append(
            BootstrapCI(
                metric="Accuracy",
                mean=round(float(bs["accuracy"]["point_estimate"]), 4),
                lower_95=round(float(bs["accuracy"]["ci_lower"]), 4),
                upper_95=round(float(bs["accuracy"]["ci_upper"]), 4),
            )
        )
    if "macro_f1" in bs:
        bootstrap_cis.append(
            BootstrapCI(
                metric="Macro F1",
                mean=round(float(bs["macro_f1"]["point_estimate"]), 4),
                lower_95=round(float(bs["macro_f1"]["ci_lower"]), 4),
                upper_95=round(float(bs["macro_f1"]["ci_upper"]), 4),
            )
        )

    loso_folds = []
    loso = data.get("loso", {})
    for idx, fld in enumerate(loso.get("fold_results", [])):
        loso_folds.append(
            LOSOFold(
                fold=idx + 1,
                test_subject=int(fld["held_out_subject"]),
                accuracy=round(float(fld["accuracy"]), 4),
                macro_f1=round(float(fld["macro_f1"]), 4),
            )
        )

    return EvaluationSummaryResponse(
        test_participants=TEST_SUBJECTS,
        total_test_sequences=int(data["dataset_info"]["test_sequences"]),
        total_time_steps=int(test_metrics["total_samples"]),
        accuracy=round(float(test_metrics["accuracy"]), 4),
        macro_f1=round(float(test_metrics["macro_f1"]), 4),
        weighted_f1=round(float(test_metrics["weighted_f1"]), 4),
        expected_calibration_error=round(float(data.get("calibration", {}).get("ece", 0.0)), 6),
        mean_sequence_log_likelihood=round(
            float(data.get("likelihood_stats", {}).get("mean_log_likelihood", 0.0)), 2
        ),
        log_likelihood_std=round(float(data.get("likelihood_stats", {}).get("std_log_likelihood", 0.0)), 2),
        per_class=per_class_dict,
        confusion_matrix=cm_data,
        calibration_bins=calib_bins,
        bootstrap_cis=bootstrap_cis,
        loso_folds=loso_folds,
        loso_mean_accuracy=round(float(loso.get("mean_accuracy", 1.0)), 4),
        loso_std_accuracy=round(float(loso.get("std_accuracy", 0.0)), 4),
    )
