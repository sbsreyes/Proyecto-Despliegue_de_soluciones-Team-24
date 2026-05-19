"""
app/api/endpoints/health.py
===========================
Endpoint de health check para monitoreo.
"""

from fastapi import APIRouter

from app.schemas.predict import HealthResponse
from app.core.config import settings
from model import __version__ as model_version

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifica que la API y el modelo estén operativos.",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=model_version,
        api_version=settings.API_VERSION,
    )
