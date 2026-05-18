"""
app/schemas/predict.py
======================
Schemas Pydantic para request/response del endpoint /predict.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# ENTRADA
# ============================================================

class DengueInput(BaseModel):
    """
    Representa UN punto de predicción (municipio + semana específica).

    Campos mínimos requeridos:
      - cod_municipio
      - anio
      - semana_epi

    El resto es opcional: si no se provee, el modelo usa los datos
    históricos del municipio almacenados en el paquete.
    """

    cod_municipio: str = Field(
        ...,
        description="Código DIVIPOLA del municipio (5 dígitos, ej: '05001' para Medellín)",
        example="05001",
        min_length=5,
        max_length=5,
    )
    anio: int = Field(
        ...,
        ge=2009, le=2030,
        description="Año de la predicción",
        example=2024,
    )
    semana_epi: int = Field(
        ...,
        ge=1, le=53,
        description="Semana epidemiológica (1-52)",
        example=32,
    )

    # ── Opcionales (si no vienen, se usan defaults del municipio) ──────────
    NOM_MPIO: Optional[str] = Field(default=None, example="MEDELLÍN")
    NOM_DPTO: Optional[str] = Field(default=None, example="ANTIOQUIA")

    altitud_msnm: Optional[float] = Field(
        default=None, ge=0, le=5000,
        description="Altitud en metros sobre nivel del mar",
        example=1523.0,
    )
    cat_altitud: Optional[str] = Field(
        default=None,
        description="Categoría altitud: 'Bajo (<1.000 m)', 'Medio (1.000-1.800 m)', "
                    "'Alto (1.800-2.200 m)', 'Muy alto (>2.200 m)'",
        example="Medio (1.000-1.800 m)",
    )

    poblacion: Optional[float] = Field(
        default=None, ge=0,
        description="Población del municipio según proyección DANE",
        example=2351077.0,
    )

    # Clima actual
    temp_media_c: Optional[float] = Field(
        default=None, ge=0, le=50,
        description="Temperatura media semanal en °C",
        example=22.5,
    )
    humedad_pct: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Humedad relativa media semanal en %",
        example=78.0,
    )
    precip_mm: Optional[float] = Field(
        default=None, ge=0,
        description="Precipitación acumulada semanal en mm",
        example=45.3,
    )

    class Config:
        schema_extra = {
            "example": {
                "cod_municipio": "05001",
                "anio": 2024,
                "semana_epi": 32,
                "NOM_MPIO": "MEDELLÍN",
                "NOM_DPTO": "ANTIOQUIA",
                "altitud_msnm": 1523.0,
                "cat_altitud": "Medio (1.000-1.800 m)",
                "poblacion": 2351077.0,
                "temp_media_c": 22.5,
                "humedad_pct": 78.0,
                "precip_mm": 45.3,
            }
        }


class PredictionRequest(BaseModel):
    """Body del POST /predict (batch)."""
    inputs: List[DengueInput] = Field(
        description="Lista de puntos a predecir (mínimo 1)",
        min_items=1,
    )

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "cod_municipio": "05001",
                        "anio": 2024,
                        "semana_epi": 32,
                        "altitud_msnm": 1523.0,
                        "cat_altitud": "Medio (1.000-1.800 m)",
                        "poblacion": 2351077.0,
                        "temp_media_c": 22.5,
                        "humedad_pct": 78.0,
                        "precip_mm": 45.3,
                    }
                ]
            }
        }


# ============================================================
# SALIDA
# ============================================================

class PredictionResult(BaseModel):
    """Resultado de UNA predicción."""
    index: int = Field(description="Índice del input correspondiente")
    cod_municipio: str
    anio: int
    semana_epi: int
    prediction: int = Field(description="0=NORMAL, 1=ALERTA")
    probability: float = Field(description="Probabilidad de alerta [0, 1]")
    label: str = Field(description="'ALERTA' o 'NORMAL'")


class PredictionResponse(BaseModel):
    """Respuesta del POST /predict."""
    predictions: List[PredictionResult]
    model_version: str
    total_records: int
    errors: Optional[str] = None


class HealthResponse(BaseModel):
    """Respuesta del GET /health."""
    status: str = Field(example="ok")
    model_version: str = Field(example="0.0.1")
    api_version: str = Field(example="1.0.0")
