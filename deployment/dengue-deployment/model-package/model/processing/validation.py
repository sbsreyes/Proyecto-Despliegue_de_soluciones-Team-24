"""
model/processing/validation.py
==============================
Valida los inputs antes de que entren al pipeline.

Detecta errores comunes:
  - tipos incorrectos
  - valores fuera de rango (ej: temperatura negativa imposible)
  - columnas faltantes
"""

from typing import List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, ValidationError


class DengueInputSchema(BaseModel):
    """
    Schema de UNA fila de input.

    Todos los campos son opcionales en Pydantic, pero
    cod_municipio + anio + semana_epi son required en la API.
    El resto puede venir nulo y XGBoost lo maneja.
    """
    cod_municipio: Optional[str] = None
    anio: Optional[int] = None
    semana_epi: Optional[int] = None
    NOM_MPIO: Optional[str] = None
    NOM_DPTO: Optional[str] = None
    altitud_msnm: Optional[float] = None
    cat_altitud: Optional[str] = None
    poblacion: Optional[float] = None
    temp_media_c: Optional[float] = None
    humedad_pct: Optional[float] = None
    precip_mm: Optional[float] = None
    temp_lag2: Optional[float] = None
    temp_lag3: Optional[float] = None
    temp_lag4: Optional[float] = None
    precip_lag2: Optional[float] = None
    precip_lag3: Optional[float] = None
    precip_lag4: Optional[float] = None
    humedad_lag1: Optional[float] = None
    humedad_lag2: Optional[float] = None
    humedad_lag3: Optional[float] = None
    casos: Optional[float] = None  # opcional: para training sí, para inferencia no
    tipo_dengue: Optional[str] = None


def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Valida un DataFrame fila por fila.

    Returns
    -------
    validated_data : pd.DataFrame
    errors : str | None
    """
    errors = None
    try:
        # Valida cada fila como un objeto DengueInputSchema
        records = input_data.to_dict(orient="records")
        for r in records:
            DengueInputSchema(**r)
    except ValidationError as exc:
        errors = str(exc)

    return input_data, errors
