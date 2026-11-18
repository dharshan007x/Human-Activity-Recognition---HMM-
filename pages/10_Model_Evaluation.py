"""Page 10: Model Evaluation - Real Test Metrics, Calibration, Bootstrap CIs, and LOSO."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import STATE_NAMES
from visualizations.charts import (
    plot_confusion_matrix_chart,
    plot_loso_comparison_chart,
    plot_reliability_diagram,
)

st.title("Model Evaluation & Experimental Benchmarks")
st.caption("Quantitative validation on held-out participants (Zero Temporal Data Leakage)")

# Load real precomputed evaluation results
eval_path = Path(__file__).resolve().parent.parent / "evaluation" / "evaluation_results.json"
if not eval_path.exists():
    from evaluation.train_and_evaluate import run_full_pipeline

    eval_results = run_full_pipeline()
else:
    with open(eval_path, encoding="utf-8") as f:
        eval_results = json.load(f)

test_m = eval_results.get("test_metrics", {})
calib = eval_results.get("calibration", {})
boot = eval_results.get("bootstrap", {})
loso = eval_results.get("loso", {})
err_analysis = eval_results.get("error_analysis", {})

# Top Summary Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Test Accuracy", f"{test_m.get('accuracy', 0.0) * 100:.2f}%")
with col2:
    st.metric("Macro-Averaged F1", f"{test_m.get('macro_f1', 0.0):.4f}")
with col3:
    st.metric("Calibration Error (ECE)", f"{calib.get('ece', 0.0):.4f}")
with col4:
    st.metric(
        "LOSO Mean Accuracy",
        f"{loso.get('mean_accuracy', 0.0) * 100:.1f} ± {loso.get('std_accuracy', 0.0) * 100:.1f}%",
    )

st.markdown("### Per-Class Classification Performance")
per_class = test_m.get("per_class", {})
table_rows = []
for name in STATE_NAMES:
    c_data = per_class.get(name, {})
    table_rows.append(
        {
            "Activity State": name,
            "Precision": f"{c_data.get('precision', 0.0):.4f}",
            "Recall": f"{c_data.get('recall', 0.0):.4f}",
            "F1-Score": f"{c_data.get('f1_score', 0.0):.4f}",
            "Test Support (Samples)": c_data.get("support", 0),
        }
    )

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# Confusion Matrix and Calibration Diagram
col_cm, col_cal = st.columns(2)
with col_cm:
    st.markdown("### Normalized Confusion Matrix")
    cm_raw = test_m.get("confusion_matrix_raw", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cm_norm = test_m.get("confusion_matrix_normalized", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    fig_cm = plot_confusion_matrix_chart(cm_raw, cm_norm, state_names=STATE_NAMES)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cal:
    st.markdown("### Reliability Diagram (Posterior Calibration)")
    fig_cal = plot_reliability_diagram(calib)
    st.plotly_chart(fig_cal, use_container_width=True)

# Bootstrap 95% Confidence Intervals
st.markdown("### Non-Parametric Bootstrap 95% Confidence Intervals (1,000 Resamples)")
boot_acc = boot.get("accuracy", {})
boot_f1 = boot.get("macro_f1", {})

boot_df = pd.DataFrame(
    [
        {
            "Metric": "Classification Accuracy",
            "Point Estimate": f"{boot_acc.get('point_estimate', 0.0) * 100:.2f}%",
            "95% CI Lower": f"{boot_acc.get('ci_lower', 0.0) * 100:.2f}%",
            "95% CI Upper": f"{boot_acc.get('ci_upper', 0.0) * 100:.2f}%",
            "Bootstrap Std": f"{boot_acc.get('std', 0.0) * 100:.2f}%",
        },
        {
            "Metric": "Macro-Averaged F1 Score",
            "Point Estimate": f"{boot_f1.get('point_estimate', 0.0):.4f}",
            "95% CI Lower": f"{boot_f1.get('ci_lower', 0.0):.4f}",
            "95% CI Upper": f"{boot_f1.get('ci_upper', 0.0):.4f}",
            "Bootstrap Std": f"{boot_f1.get('std', 0.0):.4f}",
        },
    ]
)
st.dataframe(boot_df, use_container_width=True, hide_index=True)

# Leave-One-Subject-Out Cross-Validation
st.markdown("### Leave-One-Subject-Out (LOSO) Cross-Validation Across Folds")
fig_loso = plot_loso_comparison_chart(loso)
st.plotly_chart(fig_loso, use_container_width=True)

# Biomechanical Error Analysis
st.markdown("### Biomechanical Error Analysis")
err_rows = []
for k, v in err_analysis.items():
    err_rows.append(
        {
            "Error Type": k.replace("_", " ").title(),
            "Frequency Count": v.get("count", 0),
            "Rate": f"{v.get('rate', 0.0) * 100:.2f}%",
            "Physical Biomechanical Explanation": v.get("physical_explanation", ""),
        }
    )
st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)
