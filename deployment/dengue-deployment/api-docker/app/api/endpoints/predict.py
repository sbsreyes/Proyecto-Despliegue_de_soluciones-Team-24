"""
app/api/endpoints/predict.py
=============================
Endpoints de predicción.

  POST /predict          → batch (1 o N municipios)
  POST /predict/single   → simplificado para el dashboard
"""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    DengueInput,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from model.predict import make_prediction
from model import __version__ as model_version

logger = logging.getLogger(__name__)
router = APIRouter()


def _input_to_dict(item: DengueInput) -> dict:
    """Convierte DengueInput Pydantic → dict para el modelo."""
    return item.dict(exclude_none=False)


# ────────────────────────────────────────────────────────────────────────────
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predecir alerta de dengue (batch)",
    description="Recibe uno o más puntos (municipio, año, semana) y devuelve "
                "la predicción de alerta para cada uno.",
    tags=["Prediction"],
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    logger.info(f"Predicción para {len(payload.inputs)} punto(s)")

    records = [_input_to_dict(item) for item in payload.inputs]

    try:
        result = make_prediction(input_data=records)
    except Exception as exc:
        logger.error(f"Error en make_prediction: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del modelo: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación: {result['errors']}",
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

    n_alertas = sum(p.prediction for p in prediction_results)
    logger.info(f"Resultados: {n_alertas}/{len(prediction_results)} alertas")

    return PredictionResponse(
        predictions=prediction_results,
        model_version=model_version,
        total_records=len(prediction_results),
        errors=None,
    )


# ────────────────────────────────────────────────────────────────────────────
@router.post(
    "/predict/single",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Predecir alerta de dengue (un punto)",
    description="Versión simplificada: recibe un único objeto, devuelve "
                "directamente la predicción. Ideal para el dashboard.",
    tags=["Prediction"],
)
def predict_single(item: DengueInput) -> PredictionResult:
    logger.info(f"Predicción individual: {item.cod_municipio} {item.anio} S{item.semana_epi}")

    record = _input_to_dict(item)

    try:
        result = make_prediction(input_data=[record])
    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(exc)}",
        )

    if result.get("errors"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación: {result['errors']}",
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
