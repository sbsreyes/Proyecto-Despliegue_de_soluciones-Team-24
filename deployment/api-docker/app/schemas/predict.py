"""
app/schemas/predict.py
======================
Schemas Pydantic para los endpoints de predicción.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# MODO HISTÓRICO (lo que ya existía)
# ============================================================

class DengueInput(BaseModel):
    """Input mínimo: solo identificadores. API busca todo en histórico."""
    cod_municipio: str = Field(..., min_length=5, max_length=5,
                                example="05001")
    anio: int = Field(..., ge=2009, le=2030, example=2024)
    semana_epi: int = Field(..., ge=1, le=53, example=32)

    # Opcionales: si vienen, se usan; si no, se buscan en histórico
    NOM_MPIO: Optional[str] = None
    NOM_DPTO: Optional[str] = None
    altitud_msnm: Optional[float] = Field(default=None, ge=0, le=5000)
    cat_altitud: Optional[str] = None
    poblacion: Optional[float] = Field(default=None, ge=0)
    temp_media_c: Optional[float] = Field(default=None, ge=0, le=50)
    humedad_pct: Optional[float] = Field(default=None, ge=0, le=100)
    precip_mm: Optional[float] = Field(default=None, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "cod_municipio": "05001",
                "anio": 2024,
                "semana_epi": 32,
            }
        }


class PredictionRequest(BaseModel):
    """Body batch del modo histórico."""
    inputs: List[DengueInput] = Field(..., min_items=1)


# ============================================================
# MODO REALTIME (nuevo, para dashboard)
# ============================================================

class DengueInputRealtime(BaseModel):
    """
    Input para predicción en tiempo real / simulación de escenarios.

    Usuario provee:
      - Identificadores (municipio + semana)
      - Casos recientes (lo que ve en su data)
      - Clima actual (típicamente viene del dashboard tras VLOOKUP)

    La API NO consulta histórico. Solo busca catálogo del municipio.
    """
    # Identificadores
    cod_municipio: str = Field(..., min_length=5, max_length=5,
                                description="Código DIVIPOLA (5 dígitos)",
                                example="05001")
    anio: int = Field(..., ge=2009, le=2030, example=2025)
    semana_epi: int = Field(..., ge=1, le=53, example=32)

    # Casos (required: el usuario los conoce, son la señal más importante)
    casos_actual: float = Field(..., ge=0,
                                 description="Casos en la semana actual",
                                 example=5)
    casos_lag1: float = Field(..., ge=0,
                               description="Casos hace 1 semana",
                               example=3)
    casos_lag2: float = Field(..., ge=0,
                               description="Casos hace 2 semanas",
                               example=2)
    casos_lag3: float = Field(..., ge=0,
                               description="Casos hace 3 semanas",
                               example=1)
    casos_lag4: Optional[float] = Field(default=None, ge=0,
                                         description="Casos hace 4 semanas (opcional, usa lag3 si falta)")

    # Clima (required: viene del dashboard pre-cargado, editable)
    temp_media_c: float = Field(..., ge=0, le=50,
                                 description="Temperatura media semanal (°C)",
                                 example=23.4)
    humedad_pct: float = Field(..., ge=0, le=100,
                                description="Humedad relativa (%)",
                                example=77.0)
    precip_mm: float = Field(..., ge=0,
                              description="Precipitación acumulada semanal (mm)",
                              example=145.0)

    class Config:
        schema_extra = {
            "example": {
                "cod_municipio": "05001",
                "anio": 2025,
                "semana_epi": 32,
                "casos_actual": 5,
                "casos_lag1": 3,
                "casos_lag2": 2,
                "casos_lag3": 1,
                "temp_media_c": 23.4,
                "humedad_pct": 77.0,
                "precip_mm": 145.0,
            }
        }


# ============================================================
# RESPUESTAS
# ============================================================

class PredictionResult(BaseModel):
    """Resultado de una predicción."""
    index: int
    cod_municipio: str
    anio: int
    semana_epi: int
    prediction: int = Field(description="0=NORMAL, 1=ALERTA")
    probability: float = Field(description="Probabilidad de alerta [0, 1]")
    label: str = Field(description="'ALERTA' o 'NORMAL'")


class PredictionResponse(BaseModel):
    """Respuesta batch."""
    predictions: List[PredictionResult]
    model_version: str
    total_records: int
    errors: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    api_version: str
