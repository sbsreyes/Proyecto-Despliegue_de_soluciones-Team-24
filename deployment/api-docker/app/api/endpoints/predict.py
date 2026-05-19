"""
app/api/endpoints/predict.py
=============================
Tres endpoints de predicción:

  POST /predict                  → batch HISTÓRICO (legacy)
  POST /predict/single           → single HISTÓRICO (legacy)
  POST /predict/realtime         → NUEVO: tiempo real para dashboard
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    DengueInput,
    DengueInputRealtime,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from model.predict import (
    make_prediction_historical,
    make_prediction_realtime,
)
from model import __version__ as model_version

logger = logging.getLogger(__name__)
router = APIRouter()


def _input_to_dict(item: DengueInput) -> dict:
    return item.dict(exclude_none=False)


# ────────────────────────────────────────────────────────────────────────────
# MODO HISTÓRICO — BATCH
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predicción HISTÓRICA — batch",
    description="Modo retrospectivo. Solo requiere identificadores; la API "
                "consulta histórico para construir todos los features. "
                "Útil para reproducir lo que el modelo veía en un momento dado.",
    tags=["Predicción Histórica"],
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    logger.info(f"[HIST-batch] {len(payload.inputs)} punto(s)")

    records = [_input_to_dict(item) for item in payload.inputs]

    try:
        result = make_prediction_historical(input_data=records)
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validación: {result['errors']}",
        )

    prediction_results = []
    for idx, (item, pred, prob) in enumerate(
        zip(payload.inputs, result["predictions"], result["probabilities"])
    ):
        prediction_results.append(
            PredictionResult(
                index=idx,
                cod_municipio=item.cod_municipio,
                anio=item.anio,
                semana_epi=item.semana_epi,
                prediction=pred,
                probability=prob,
                label="ALERTA" if pred == 1 else "NORMAL",
            )
        )

    return PredictionResponse(
        predictions=prediction_results,
        model_version=model_version,
        total_records=len(prediction_results),
        errors=None,
    )


# ────────────────────────────────────────────────────────────────────────────
# MODO HISTÓRICO — SINGLE
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict/single",
    response_model=PredictionResult,
    summary="Predicción HISTÓRICA — single",
    description="Versión single del predict histórico.",
    tags=["Predicción Histórica"],
)
def predict_single(item: DengueInput) -> PredictionResult:
    logger.info(f"[HIST-single] {item.cod_municipio} {item.anio}S{item.semana_epi}")

    record = _input_to_dict(item)

    try:
        result = make_prediction_historical(input_data=[record])
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validación: {result['errors']}",
        )

    pred = result["predictions"][0]
    prob = result["probabilities"][0]

    return PredictionResult(
        index=0,
        cod_municipio=item.cod_municipio,
        anio=item.anio,
        semana_epi=item.semana_epi,
        prediction=pred,
        probability=prob,
        label="ALERTA" if pred == 1 else "NORMAL",
    )


# ────────────────────────────────────────────────────────────────────────────
# MODO REALTIME — para el dashboard
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict/realtime",
    response_model=PredictionResult,
    summary="Predicción TIEMPO REAL — para dashboard",
    description=(
        "Modo prospectivo / simulación de escenarios. Usuario provee:\n"
        "- Identificadores (municipio + semana)\n"
        "- Casos recientes (actual + 3 lags)\n"
        "- Clima actual (típicamente pre-cargado desde catálogo en dashboard)\n\n"
        "La API NO consulta histórico de casos. Catálogo del municipio "
        "(altitud, población) se busca automáticamente. Lags climáticos se "
        "asumen iguales al clima actual."
    ),
    tags=["Predicción Tiempo Real"],
)
def predict_realtime(item: DengueInputRealtime) -> PredictionResult:
    logger.info(
        f"[RT] {item.cod_municipio} {item.anio}S{item.semana_epi} "
        f"casos={item.casos_actual} temp={item.temp_media_c}"
    )

    record = item.dict()

    try:
        result = make_prediction_realtime(input_data=[record])
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validación: {result['errors']}",
        )

    pred = result["predictions"][0]
    prob = result["probabilities"][0]

    return PredictionResult(
        index=0,
        cod_municipio=item.cod_municipio,
        anio=item.anio,
        semana_epi=item.semana_epi,
        prediction=pred,
        probability=prob,
        label="ALERTA" if pred == 1 else "NORMAL",
    )
