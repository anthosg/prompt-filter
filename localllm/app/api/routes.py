from fastapi import APIRouter

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference.generator import generate_text
from app.inference.model_loader import get_model_info

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="localllm")


@router.get("/v1/models", response_model=ModelInfoResponse)
def models() -> ModelInfoResponse:
    return get_model_info()


@router.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    return generate_text(request)
