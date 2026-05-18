"""
model/predict.py
================
Función de inferencia consumida por la API.
 
Uso:
    from model.predict import make_prediction
 
    result = make_prediction(input_data=[{
        "cod_municipio": "05001",
        "anio": 2024,
        "semana_epi": 32,
        "temp_media_c": 24.5,
        "humedad_pct": 78,
        "precip_mm": 145.2,
        "altitud_msnm": 1523,
        "cat_altitud": "Medio (1.000-1.800 m)",
        "poblacion": 2351077,
        # ... opcionalmente más campos
    }])
 
    # result["predictions"]   → [1]
    # result["probabilities"] → [0.87]
    # result["version"]       → "0.0.1"
 
Si el input está incompleto (faltan lags, por ejemplo), la función
automáticamente consulta el histórico del municipio en el CSV local
para construir los lags antes de predecir.
"""
 
import typing as t
import numpy as np
import pandas as pd
 
from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import (
    load_pipeline,
    get_municipio_context,
)
from model.processing.validation import validate_inputs
 
 
# ── Cargar el pipeline una sola vez al importar ─────────────────────────────
_pipeline_file = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_dengue_pipe = load_pipeline(file_name=_pipeline_file)
 
 
def _enrich_with_history(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fila de inferencia, busca el histórico del municipio
    y prepara un DataFrame con suficiente contexto para que el
    FeatureEngineer pueda calcular lags.
 
    Si el request no provee algún campo (ej: poblacion, altitud_msnm),
    se rellena con el último valor conocido del histórico del municipio.
    """
    # Columnas que se pueden tomar del histórico si no vienen en el request
    FILLABLE_FROM_HISTORY = [
        "altitud_msnm", "cat_altitud", "poblacion",
        "NOM_MPIO", "NOM_DPTO",
        "temp_media_c", "humedad_pct", "precip_mm",
        "temp_lag2", "temp_lag3", "temp_lag4",
        "precip_lag2", "precip_lag3", "precip_lag4",
        "humedad_lag1", "humedad_lag2", "humedad_lag3",
    ]
 
    # Columnas que deben ser numéricas (force-cast antes de XGBoost)
    NUMERIC_COLS = [
        "anio", "semana_epi", "altitud_msnm", "poblacion",
        "temp_media_c", "humedad_pct", "precip_mm",
        "temp_lag2", "temp_lag3", "temp_lag4",
        "precip_lag2", "precip_lag3", "precip_lag4",
        "humedad_lag1", "humedad_lag2", "humedad_lag3",
        "casos",
    ]
 
    enriched_rows = []
 
    for _, row in input_df.iterrows():
        cod_mun = row["cod_municipio"]
        anio = int(row["anio"])
        sem = int(row["semana_epi"])
 
        # Histórico previo a la semana objetivo
        hist = get_municipio_context(cod_mun, anio, sem, n_weeks_back=8)
 
        # Construir fila nueva con los valores del request
        new_row = row.to_dict()
        # Rellenar campos faltantes con el último valor del histórico
        if len(hist) > 0:
            last = hist.iloc[-1]
            for col in FILLABLE_FROM_HISTORY:
                if new_row.get(col) is None and col in hist.columns:
                    new_row[col] = last[col]
 
        # Asegurar `casos` (en inferencia siempre es 0 porque es lo que predecimos)
        new_row.setdefault("casos", 0)
        if new_row.get("casos") is None:
            new_row["casos"] = 0
 
        # Combinar histórico + fila nueva
        combined = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
 
        # Force-cast a numérico para evitar columns object que rompen XGBoost
        for col in NUMERIC_COLS:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
 
        enriched_rows.append(combined)
 
    full_df = pd.concat(enriched_rows, ignore_index=True)
    return full_df
 
 
def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, t.List[dict], dict],
) -> dict:
    """
    Realiza predicciones de alerta de dengue.
 
    Args:
        input_data: dict, lista de dicts o DataFrame con columnas crudas
                    Mínimo requerido: cod_municipio, anio, semana_epi.
 
    Returns:
        dict con:
          - predictions: lista de 0 (NORMAL) o 1 (ALERTA)
          - probabilities: probabilidad de alerta [0.0, 1.0]
          - version: versión del modelo
          - errors: None o mensaje de error
    """
    # Normalizar input
    if isinstance(input_data, dict):
        input_data = [input_data]
    data = pd.DataFrame(input_data)
 
    # Validar
    validated_data, errors = validate_inputs(input_data=data)
    if errors:
        return {
            "predictions": None,
            "probabilities": None,
            "version": _version,
            "errors": errors,
        }
 
    # Enriquecer con histórico del municipio para construir lags
    try:
        enriched = _enrich_with_history(validated_data)
    except Exception as exc:
        return {
            "predictions": None,
            "probabilities": None,
            "version": _version,
            "errors": f"Error enriqueciendo con histórico: {exc}",
        }
 
    # Predecir
    try:
        predictions_all = _dengue_pipe.predict(enriched)
        probabilities_all = _dengue_pipe.predict_proba(enriched)[:, 1]
    except Exception as exc:
        return {
            "predictions": None,
            "probabilities": None,
            "version": _version,
            "errors": f"Error en predict: {exc}",
        }
 
    # Solo retornamos predicciones de las filas que el usuario pasó
    # (las últimas N filas, donde N = len(validated_data))
    n_input = len(validated_data)
 
    # Tomar las últimas N predicciones de cada bloque enriquecido
    # Cada bloque tiene `historical + 1 nueva` rows, así que la predicción
    # de cada input es la fila final de cada grupo
    preds = []
    probs = []
    idx = 0
    for _, row in validated_data.iterrows():
        # Calcular tamaño del bloque para este input (hist + 1)
        cod_mun = row["cod_municipio"]
        anio = int(row["anio"])
        sem = int(row["semana_epi"])
        hist = get_municipio_context(cod_mun, anio, sem, n_weeks_back=8)
        block_size = len(hist) + 1
        # La predicción del input es la última fila del bloque
        idx_pred = idx + block_size - 1
        preds.append(int(predictions_all[idx_pred]))
        probs.append(round(float(probabilities_all[idx_pred]), 4))
        idx += block_size
 
    return {
        "predictions": preds,
        "probabilities": probs,
        "version": _version,
        "errors": None,
    }