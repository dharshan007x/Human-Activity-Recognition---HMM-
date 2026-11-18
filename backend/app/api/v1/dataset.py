"""Dataset inspection and signal endpoints."""

from fastapi import APIRouter, Query
from backend.app.services.dataset_service import DatasetService
from backend.app.schemas.dataset import (
    DatasetMetadataResponse,
    DatasetSummaryResponse,
    DatasetPreviewResponse,
    SignalStreamResponse,
)
from backend.app.schemas.inference import SampleSequenceItem

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/metadata", response_model=DatasetMetadataResponse)
def get_metadata():
    service = DatasetService.get_instance()
    return service.get_metadata()


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_summary():
    service = DatasetService.get_instance()
    return service.get_summary()


@router.get("/preview", response_model=DatasetPreviewResponse)
def get_preview(limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    service = DatasetService.get_instance()
    return service.get_preview(limit=limit, offset=offset)


@router.get("/signals", response_model=SignalStreamResponse)
def get_signals(
    subject_id: int = Query(default=101),
    activity_name: str = Query(default="Walking"),
    max_points: int = Query(default=150, ge=10, le=1000),
):
    service = DatasetService.get_instance()
    return service.get_signals(subject_id=subject_id, activity_name=activity_name, max_points=max_points)


@router.get("/filter-response")
def get_filter_response():
    service = DatasetService.get_instance()
    return service.get_butterworth_response()


@router.get("/samples", response_model=list[SampleSequenceItem])
def get_samples():
    service = DatasetService.get_instance()
    return service.get_sample_sequences()
