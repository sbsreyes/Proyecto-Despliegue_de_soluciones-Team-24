"""Tests del módulo de predicción."""

from model.predict import make_prediction
from model import __version__


def test_make_prediction_returns_correct_structure(sample_input_data):
    result = make_prediction(input_data=sample_input_data)

    assert "predictions" in result
    assert "probabilities" in result
    assert "version" in result
    assert "errors" in result
    assert result["version"] == __version__


def test_make_prediction_no_errors(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    assert result["errors"] is None
    assert result["predictions"] is not None
    assert len(result["predictions"]) == 1


def test_make_prediction_values_in_range(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    # Predicción debe ser 0 o 1
    assert result["predictions"][0] in [0, 1]
    # Probabilidad debe estar en [0, 1]
    assert 0.0 <= result["probabilities"][0] <= 1.0


def test_make_prediction_batch():
    inputs = [
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
        },
        {
            "cod_municipio": "76001",
            "anio": 2024,
            "semana_epi": 32,
            "altitud_msnm": 995.0,
            "cat_altitud": "Bajo (<1.000 m)",
            "poblacion": 2228000.0,
            "temp_media_c": 25.0,
            "humedad_pct": 75.0,
            "precip_mm": 80.0,
        },
    ]
    result = make_prediction(input_data=inputs)
    assert result["errors"] is None
    assert len(result["predictions"]) == 2
    assert len(result["probabilities"]) == 2
