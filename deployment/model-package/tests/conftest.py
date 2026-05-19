import pytest
import pandas as pd

from model.config.core import config
from model.processing.data_manager import load_dataset, aggregate_by_municipio


@pytest.fixture()
def sample_input_data():
    """Datos de ejemplo para un municipio (Medellín)."""
    return [{
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
    }]


@pytest.fixture()
def raw_training_data():
    return load_dataset(file_name=config.app_config.train_data_file)
