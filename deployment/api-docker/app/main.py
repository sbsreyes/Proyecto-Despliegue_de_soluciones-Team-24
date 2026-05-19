"""
app/main.py
===========
Punto de entrada de la API FastAPI.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.endpoints import health, predict as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict_router.router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 60)
    logger.info("AI-lerta Dengue Colombia API — Arrancando")
    logger.info(f"Versión API: {settings.API_VERSION}")

    try:
        from model import __version__ as model_version
        from model.predict import _dengue_pipe   # fuerza carga del .pkl
        logger.info(f"Versión modelo: {model_version}")
        logger.info(f"Pipeline cargado: {[s for s, _ in _dengue_pipe.steps]}")
        logger.info("✓ Modelo listo para inferencia")
    except Exception as e:
        logger.error(f"✗ ERROR cargando modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")

    logger.info(f"Docs: http://0.0.0.0:{settings.PORT}/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("API apagándose.")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
