"""
model/config/core.py
====================
Carga config.yml y lo valida con Pydantic.

Usa PyYAML (estándar) en lugar de strictyaml para evitar
restricciones sobre flow sequences y casteos manuales de tipos.

Acceso a la configuración:
    from model.config.core import config
    config.app_config.train_data_file       # → "dataset_dengue_completo.csv"
    config.model_config_.xgb_max_depth      # → 6
    config.model_config_.model_features     # → lista de 35 features
"""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

from model import PACKAGE_ROOT


# ── Paths importantes ────────────────────────────────────────────────────────
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


# ── Schemas Pydantic ─────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Configuración de paths y archivos."""
    package_name: str
    train_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """Hiperparámetros del modelo y feature engineering."""
    target: str

    # Splits
    train_years: List[int]
    validation_years: List[int]
    test_years: List[int]
    canal_endemico_years: List[int]
    min_semanas_con_casos: int

    # Columnas
    raw_columns: List[str]
    model_features: List[str]

    # Categorías
    cat_altitud_categorias: List[str]

    # Años epidémicos
    anos_epidemicos: List[int]
    anos_post_epidemia: List[int]

    # Lluvias
    temporada_lluvias_1: List[int]
    temporada_lluvias_2: List[int]

    # Biología Aedes
    temp_min_desarrollo: int
    temp_optima_min: int
    temp_optima_max: int
    temp_letal: int
    temp_inhibicion: int
    altitud_inhibicion: int

    # XGBoost
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_scale_pos_weight: int
    random_state: int

    umbral_default: float


class Config(BaseModel):
    """Master config object."""
    app_config: AppConfig
    model_config_: ModelConfig


# ── Funciones de carga ───────────────────────────────────────────────────────

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"No se encontró config.yml en {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> dict:
    """Carga el config.yml como dict de Python usando PyYAML."""
    if cfg_path is None:
        cfg_path = find_config_file()
    with open(cfg_path, "r", encoding="utf-8") as conf_file:
        parsed_config = yaml.safe_load(conf_file)
    return parsed_config


def create_and_validate_config(parsed_config: dict = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_config_=ModelConfig(**parsed_config),
    )
    return _config


# ── Instancia única exportada ────────────────────────────────────────────────
config = create_and_validate_config()
