"""
app/core/config.py
==================
Configuración centralizada de la API.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # ── Identidad ────────────────────────────────────────────────────────────
    API_TITLE: str = "AI-lerta Dengue Colombia API"
    API_DESCRIPTION: str = (
        "Sistema de Alerta Temprana de Dengue para Colombia. "
        "Predice si un municipio entrará en alerta epidemiológica "
        "para una semana epidemiológica específica, basado en variables "
        "climáticas (CHIRPS + ERA5), biológicas del vector Aedes aegypti, "
        "y el canal endémico municipal histórico. "
        "Modelo: XGBoost — F2=0.78 AUC=0.87 en test 2022-2024."
    )
    API_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8001")))

    # ── CORS ────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        "*",
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit
        "http://localhost:8050",  # Dash
    ])

    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"


settings = Settings()
