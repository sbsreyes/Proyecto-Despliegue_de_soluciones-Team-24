"""
model/pipeline.py
=================
Define el sklearn.Pipeline completo del modelo de dengue.

Arquitectura:
    DengueFeatureEngineer  →  XGBClassifier

- DengueFeatureEngineer: replica todo el feature engineering del notebook
  (variables biológicas, lags, estacionalidad, índice de idoneidad, etc.)
- XGBClassifier: clasificador binario (alerta=1 si casos >= P75 histórico)

Hiperparámetros del notebook 05_modelo_dengue_FINAL150526.ipynb (mejor modelo).
"""

from sklearn.pipeline import Pipeline
import xgboost as xgb

from model.config.core import config
from model.processing.features import DengueFeatureEngineer


dengue_pipeline = Pipeline([
    (
        "feature_engineer",
        DengueFeatureEngineer(
            config_dict={
                "cat_altitud_categorias": config.model_config_.cat_altitud_categorias,
                "model_features": config.model_config_.model_features,
                "anos_epidemicos": config.model_config_.anos_epidemicos,
                "anos_post_epidemia": config.model_config_.anos_post_epidemia,
                "temporada_lluvias_1": config.model_config_.temporada_lluvias_1,
                "temporada_lluvias_2": config.model_config_.temporada_lluvias_2,
                "temp_min_desarrollo": config.model_config_.temp_min_desarrollo,
                "temp_optima_min": config.model_config_.temp_optima_min,
                "temp_optima_max": config.model_config_.temp_optima_max,
                "temp_letal": config.model_config_.temp_letal,
                "temp_inhibicion": config.model_config_.temp_inhibicion,
                "altitud_inhibicion": config.model_config_.altitud_inhibicion,
            }
        ),
    ),
    (
        "classifier",
        xgb.XGBClassifier(
            n_estimators=config.model_config_.xgb_n_estimators,
            max_depth=config.model_config_.xgb_max_depth,
            learning_rate=config.model_config_.xgb_learning_rate,
            subsample=config.model_config_.xgb_subsample,
            colsample_bytree=config.model_config_.xgb_colsample_bytree,
            scale_pos_weight=config.model_config_.xgb_scale_pos_weight,
            eval_metric="logloss",
            random_state=config.model_config_.random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    ),
])
