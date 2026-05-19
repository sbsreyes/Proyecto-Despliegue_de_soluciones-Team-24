"""
model/predict.py
================
Dos funciones de inferencia según el caso de uso:

1. make_prediction_historical(input_data)
   Modo retrospectivo. Usuario solo da identificadores; la función consulta
   el CSV histórico embebido para construir TODOS los features.

2. make_prediction_realtime(input_data)
   Modo tiempo real / simulación. Usuario provee casos recientes y clima.
   La función NO consulta histórico de casos. Solo busca catálogo del
   municipio (altitud, cat_altitud, población) del CSV embebido.
"""

import typing as t
import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import (
    load_pipeline,
    get_municipio_context,
    get_catalogo_municipio,
)
from model.processing.validation import (
    validate_inputs_historical,
    validate_inputs_realtime,
)


# ── Cargar el pipeline una sola vez al importar ─────────────────────────────
_pipeline_file = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_dengue_pipe = load_pipeline(file_name=_pipeline_file)


NUMERIC_COLS = [
    "anio", "semana_epi", "altitud_msnm", "poblacion",
    "temp_media_c", "humedad_pct", "precip_mm",
    "temp_lag2", "temp_lag3", "temp_lag4",
    "precip_lag2", "precip_lag3", "precip_lag4",
    "humedad_lag1", "humedad_lag2", "humedad_lag3",
    "casos",
]


def _force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas numéricas a float."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ────────────────────────────────────────────────────────────────────────────
# MODO 1: HISTÓRICO
# ────────────────────────────────────────────────────────────────────────────

def _enrich_from_history(input_df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece cada input con el histórico del municipio para calcular lags."""
    FILLABLE_FROM_HISTORY = [
        "altitud_msnm", "cat_altitud", "poblacion",
        "NOM_MPIO", "NOM_DPTO",
        "temp_media_c", "humedad_pct", "precip_mm",
        "temp_lag2", "temp_lag3", "temp_lag4",
        "precip_lag2", "precip_lag3", "precip_lag4",
        "humedad_lag1", "humedad_lag2", "humedad_lag3",
    ]

    enriched_rows = []
    for _, row in input_df.iterrows():
        cod_mun = row["cod_municipio"]
        anio = int(row["anio"])
        sem = int(row["semana_epi"])

        hist = get_municipio_context(cod_mun, anio, sem, n_weeks_back=8)

        new_row = row.to_dict()
        if len(hist) > 0:
            last = hist.iloc[-1]
            for col in FILLABLE_FROM_HISTORY:
                if new_row.get(col) is None and col in hist.columns:
                    new_row[col] = last[col]

        new_row.setdefault("casos", 0)
        if new_row.get("casos") is None:
            new_row["casos"] = 0

        combined = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        combined = _force_numeric(combined)
        enriched_rows.append(combined)

    return pd.concat(enriched_rows, ignore_index=True)


def make_prediction_historical(
    *,
    input_data: t.Union[pd.DataFrame, t.List[dict], dict],
) -> dict:
    """Predicción retrospectiva: solo requiere cod_municipio, anio, semana_epi."""
    if isinstance(input_data, dict):
        input_data = [input_data]
    data = pd.DataFrame(input_data)

    validated, errors = validate_inputs_historical(input_data=data)
    if errors:
        return {"predictions": None, "probabilities": None,
                "version": _version, "errors": errors}

    try:
        enriched = _enrich_from_history(validated)
    except Exception as exc:
        return {"predictions": None, "probabilities": None,
                "version": _version, "errors": f"Error enriqueciendo: {exc}"}

    try:
        preds_all = _dengue_pipe.predict(enriched)
        probs_all = _dengue_pipe.predict_proba(enriched)[:, 1]
    except Exception as exc:
        return {"predictions": None, "probabilities": None,
                "version": _version, "errors": f"Error en predict: {exc}"}

    preds, probs = [], []
    idx = 0
    for _, row in validated.iterrows():
        cod_mun = row["cod_municipio"]
        anio = int(row["anio"])
        sem = int(row["semana_epi"])
        hist = get_municipio_context(cod_mun, anio, sem, n_weeks_back=8)
        block_size = len(hist) + 1
        idx_pred = idx + block_size - 1
        preds.append(int(preds_all[idx_pred]))
        probs.append(round(float(probs_all[idx_pred]), 4))
        idx += block_size

    return {"predictions": preds, "probabilities": probs,
            "version": _version, "errors": None}


# ────────────────────────────────────────────────────────────────────────────
# MODO 2: REALTIME
# ────────────────────────────────────────────────────────────────────────────

def _build_realtime_row(req: dict) -> dict:
    """Construye una fila completa para realtime con catálogo automático."""
    cod_mun = req["cod_municipio"]
    catalogo = get_catalogo_municipio(cod_mun)

    casos_actual = req.get("casos_actual", 0) or 0

    temp = req["temp_media_c"]
    humedad = req["humedad_pct"]
    precip = req["precip_mm"]

    row = {
        "cod_municipio": cod_mun,
        "anio": int(req["anio"]),
        "semana_epi": int(req["semana_epi"]),
        "tipo_dengue": "grave",
        "casos": casos_actual,

        "altitud_msnm": catalogo.get("altitud_msnm"),
        "cat_altitud": catalogo.get("cat_altitud", "Sin dato"),
        "poblacion": catalogo.get("poblacion"),
        "NOM_MPIO": catalogo.get("NOM_MPIO"),
        "NOM_DPTO": catalogo.get("NOM_DPTO"),

        "temp_media_c": temp,
        "humedad_pct": humedad,
        "precip_mm": precip,

        # Lags climáticos = repetición del actual (decisión arquitectónica)
        "temp_lag2": temp, "temp_lag3": temp, "temp_lag4": temp,
        "precip_lag2": precip, "precip_lag3": precip, "precip_lag4": precip,
        "humedad_lag1": humedad, "humedad_lag2": humedad, "humedad_lag3": humedad,
    }
    return row


def make_prediction_realtime(
    *,
    input_data: t.Union[pd.DataFrame, t.List[dict], dict],
) -> dict:
    """
    Predicción en tiempo real. Usuario provee todo el contexto:
    casos recientes + clima actual.

    NO consulta histórico. Solo busca catálogo (altitud/pob) del municipio.
    """
    if isinstance(input_data, dict):
        input_data = [input_data]
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.to_dict(orient="records")

    validated_records = []
    for req in input_data:
        errors = validate_inputs_realtime(req)
        if errors:
            return {"predictions": None, "probabilities": None,
                    "version": _version, "errors": errors}
        validated_records.append(req)

    # Construir DataFrame
    rows = [_build_realtime_row(req) for req in validated_records]
    df = pd.DataFrame(rows)
    df = _force_numeric(df)

    # Aplicar FeatureEngineer
    fe = _dengue_pipe.named_steps["feature_engineer"]
    classifier = _dengue_pipe.named_steps["classifier"]

    try:
        features_df = fe.transform(df)
    except Exception as exc:
        return {"predictions": None, "probabilities": None,
                "version": _version, "errors": f"Error en FeatureEngineer: {exc}"}

    # Sobreescribir lags de casos (FE no puede calcularlos sin histórico)
    features_df = features_df.reset_index(drop=True)
    for i, req in enumerate(validated_records):
        l1 = req.get("casos_lag1", 0) or 0
        l2 = req.get("casos_lag2", 0) or 0
        l3 = req.get("casos_lag3", 0) or 0
        l4 = req.get("casos_lag4", l3) or l3
        features_df.loc[i, "casos_lag1"] = l1
        features_df.loc[i, "casos_lag2"] = l2
        features_df.loc[i, "casos_lag3"] = l3
        features_df.loc[i, "casos_lag4"] = l4
        features_df.loc[i, "casos_ma4"] = (l1 + l2 + l3 + l4) / 4.0
        features_df.loc[i, "casos_tendencia"] = l1 - l2

    # Predict
    try:
        preds = classifier.predict(features_df)
        probs = classifier.predict_proba(features_df)[:, 1]
    except Exception as exc:
        return {"predictions": None, "probabilities": None,
                "version": _version, "errors": f"Error en predict: {exc}"}

    return {
        "predictions": [int(p) for p in preds],
        "probabilities": [round(float(p), 4) for p in probs],
        "version": _version,
        "errors": None,
    }


# ────────────────────────────────────────────────────────────────────────────
# COMPATIBILIDAD (alias para no romper código existente)
# ────────────────────────────────────────────────────────────────────────────

def make_prediction(*, input_data) -> dict:
    """Alias legacy. Delega a make_prediction_historical."""
    return make_prediction_historical(input_data=input_data)
