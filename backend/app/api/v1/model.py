"""HMM model parameter inspection endpoints."""

from fastapi import APIRouter
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/parameters")
def get_model_parameters():
    service = ModelService.get_instance()
    return service.get_parameters_dict()
