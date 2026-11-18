"""CSV upload and human activity recognition endpoint."""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import pandas as pd

from backend.app.schemas.recognize import RecognitionResponse, StepPrediction
from backend.app.services.model_service import ModelService
from backend.config import STATE_NAMES, TARGET_SAMPLING_RATE_HZ

router = APIRouter(prefix="/recognize", tags=["Recognition"])


@router.post("", response_model=RecognitionResponse)
async def recognize_csv(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload a CSV file (.csv).")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}") from e

    # Find accelerometer columns
    col_map = {}
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    for cand in ["acc_x", "ax", "accx", "acceleration_x"]:
        if cand in cols_lower:
            col_map["ax"] = cols_lower[cand]
            break
    for cand in ["acc_y", "ay", "accy", "acceleration_y"]:
        if cand in cols_lower:
            col_map["ay"] = cols_lower[cand]
            break
    for cand in ["acc_z", "az", "accz", "acceleration_z"]:
        if cand in cols_lower:
            col_map["az"] = cols_lower[cand]
            break

    if len(col_map) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            col_map["ax"] = numeric_cols[0]
            col_map["ay"] = numeric_cols[1]
            col_map["az"] = numeric_cols[2]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"CSV must contain at least 3 triaxial accelerometer columns (e.g. acc_x, acc_y, acc_z). "
                    f"Found columns: {list(df.columns)}"
                ),
            )

    sub_df = df[[col_map["ax"], col_map["ay"], col_map["az"]]].dropna()
    if sub_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains 0 valid numeric rows.")

    ax = sub_df[col_map["ax"]].to_numpy(dtype=np.float64)
    ay = sub_df[col_map["ay"]].to_numpy(dtype=np.float64)
    az = sub_df[col_map["az"]].to_numpy(dtype=np.float64)

    if np.any(np.isinf(ax)) or np.any(np.isinf(ay)) or np.any(np.isinf(az)):
        raise HTTPException(status_code=400, detail="Accelerometer data contains Infinite values.")

    amag = np.sqrt(ax**2 + ay**2 + az**2)
    raw_feats = np.column_stack([ax, ay, az, amag])

    # Truncate to reasonable max sequence length if very long (e.g. 500 timesteps)
    if len(raw_feats) > 500:
        raw_feats = raw_feats[:500]
        ax, ay, az, amag = ax[:500], ay[:500], az[:500], amag[:500]

    model_service = ModelService.get_instance()
    scaler = model_service.scaler
    hmm = model_service.hmm

    scaled_feats = scaler.transform([raw_feats])[0]
    gamma, pred_indices, fwd_ll, _, _ = hmm.posterior(scaled_feats, method="log")
    vit_path, vit_ll, _ = hmm.decode_viterbi(scaled_feats)

    steps = []
    for t in range(len(raw_feats)):
        pred_act = STATE_NAMES[pred_indices[t]]
        conf = float(np.max(gamma[t]))
        steps.append(
            StepPrediction(
                step=t,
                t_sec=round(t / TARGET_SAMPLING_RATE_HZ, 3),
                acc_x=round(float(ax[t]), 4),
                acc_y=round(float(ay[t]), 4),
                acc_z=round(float(az[t]), 4),
                acc_mag=round(float(amag[t]), 4),
                predicted_activity=pred_act,
                confidence=round(conf, 4),
                walking_prob=round(float(gamma[t, 0]), 4),
                sitting_prob=round(float(gamma[t, 1]), 4),
                running_prob=round(float(gamma[t, 2]), 4),
            )
        )

    counts = pd.Series([s.predicted_activity for s in steps]).value_counts()
    dist = {act: round(float(counts.get(act, 0) / len(steps)), 4) for act in STATE_NAMES}
    majority_act = counts.index[0] if not counts.empty else "Unknown"

    return RecognitionResponse(
        filename=file.filename,
        total_timesteps=len(steps),
        duration_seconds=round(len(steps) / TARGET_SAMPLING_RATE_HZ, 2),
        overall_predicted_activity=majority_act,
        activity_distribution=dist,
        mean_confidence=round(float(np.mean([s.confidence for s in steps])), 4),
        forward_log_likelihood=round(float(fwd_ll), 4),
        viterbi_log_likelihood=round(float(vit_ll), 4),
        steps=steps,
        status="SUCCESS",
        message=f"Successfully classified {len(steps)} time steps. Overall activity: {majority_act}.",
    )
