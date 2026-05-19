"""
model/processing/validation.py
==============================
Validadores Pydantic separados para los dos modos de inferencia.
"""

from typing import List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# ────────────────────────────────────────────────────────────────────────────
# Validador modo HISTÓRICO
# ────────────────────────────────────────────────────────────────────────────

class DengueInputHistorical(BaseModel):
    """
    Schema para modo retrospectivo.
    Solo requiere identificadores; el sistema busca el resto en histórico.
    """
    cod_municipio: str = Field(..., min_length=5, max_length=5)
    anio: int = Field(..., ge=2009, le=2030)
    semana_epi: int = Field(..., ge=1, le=53)

    # Opcionales (si vienen, los usa; si no, los busca)
    NOM_MPIO: Optional[str] = None
    NOM_DPTO: Optional[str] = None
    altitud_msnm: Optional[float] = None
    cat_altitud: Optional[str] = None
    poblacion: Optional[float] = None
    temp_media_c: Optional[float] = None
    humedad_pct: Optional[float] = None
    precip_mm: Optional[float] = None
    tipo_dengue: Optional[str] = None
    casos: Optional[float] = None


def validate_inputs_historical(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Valida cada fila como DengueInputHistorical."""
    errors = None
    try:
        for r in input_data.to_dict(orient="records"):
            DengueInputHistorical(**r)
    except ValidationError as exc:
        errors = str(exc)
    return input_data, errors


# ────────────────────────────────────────────────────────────────────────────
# Validador modo REALTIME
# ────────────────────────────────────────────────────────────────────────────

class DengueInputRealtime(BaseModel):
    """
    Schema para modo tiempo real / dashboard.
    Usuario debe proveer casos recientes + clima actual.
    """
    cod_municipio: str = Field(..., min_length=5, max_length=5)
    anio: int = Field(..., ge=2009, le=2030)
    semana_epi: int = Field(..., ge=1, le=53)

    # Casos (required)
    casos_actual: float = Field(..., ge=0)
    casos_lag1: float = Field(..., ge=0)
    casos_lag2: float = Field(..., ge=0)
    casos_lag3: float = Field(..., ge=0)
    casos_lag4: Optional[float] = Field(default=None, ge=0)

    # Clima (required)
    temp_media_c: float = Field(..., ge=0, le=50)
    humedad_pct: float = Field(..., ge=0, le=100)
    precip_mm: float = Field(..., ge=0)


def validate_inputs_realtime(req: dict) -> Optional[str]:
    """
    Valida un solo request realtime.
    Returns: None si OK, str con errores si falla.
    """
    try:
        DengueInputRealtime(**req)
    except ValidationError as exc:
        return str(exc)
    return None


# ────────────────────────────────────────────────────────────────────────────
# Alias para compatibilidad con código viejo (no romper imports)
# ────────────────────────────────────────────────────────────────────────────

DengueInputSchema = DengueInputHistorical

def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Alias legacy de validate_inputs_historical."""
    return validate_inputs_historical(input_data=input_data)
