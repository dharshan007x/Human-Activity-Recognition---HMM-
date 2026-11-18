"""Inference API endpoints executing pure NumPy Forward, Backward, and Posterior routines."""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.inference import (
    InferenceRequest,
    ForwardResponse,
    BackwardResponse,
    PosteriorResponse,
    PredictResponse,
    ValidationResponse,
)
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/inference", tags=["Inference"])
service = InferenceService()


@router.post("/forward", response_model=ForwardResponse)
def execute_forward(req: InferenceRequest):
    try:
        return service.run_forward(req.sequence, scale_input=req.scale_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/backward", response_model=BackwardResponse)
def execute_backward(req: InferenceRequest):
    try:
        return service.run_backward(req.sequence, scale_input=req.scale_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/posterior", response_model=PosteriorResponse)
def execute_posterior(req: InferenceRequest):
    try:
        return service.run_posterior(req.sequence, scale_input=req.scale_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/predict", response_model=PredictResponse)
def execute_predict(req: InferenceRequest):
    try:
        return service.predict_full(req.sequence, scale_input=req.scale_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/validate", response_model=ValidationResponse)
def validate_forward_backward(req: InferenceRequest):
    try:
        return service.validate_likelihoods(req.sequence, scale_input=req.scale_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
