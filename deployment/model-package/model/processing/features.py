"""
model/processing/features.py
============================
DengueFeatureEngineer
---------------------
Custom transformer scikit-learn que replica todo el feature engineering
del notebook 05_modelo_dengue_FINAL150526.ipynb.

Recibe:
    DataFrame con las columnas crudas (raw_columns en config.yml).

Genera:
    DataFrame con las 35 features finales (model_features en config.yml).

Pasos:
    1. Encode cat_altitud → cat_altitud_enc (LabelEncoder)
    2. Variables biológicas: grados_dia, temp_optima, temp_letal, temp_inhibicion
    3. Índice de idoneidad altitud × temperatura
    4. Precipitación acumulada 4 y 8 semanas (rolling)
    5. Estacionalidad: semana_sin, semana_cos, temporada_lluvias
    6. Años epidémicos / post-epidemia
    7. Autoregresión: casos_lag1-4, casos_ma4, casos_tendencia

NOTA: el LabelEncoder se ajusta en .fit() solo con datos de train,
y en .transform() se aplica al input nuevo. Esto evita data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class DengueFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Genera todas las features derivadas para el modelo de dengue.

    Parameters
    ----------
    config : ModelConfig
        Configuración con todos los parámetros (umbrales, años epidémicos, etc.)
    """

    def __init__(self, config_dict: dict):
        # Guardamos como dict (sklearn no se lleva bien con Pydantic en clone)
        self.config_dict = config_dict

    # ────────────────────────────────────────────────────────────────────────
    def fit(self, X, y=None):
        """
        Ajusta el LabelEncoder de cat_altitud con todas las categorías posibles.
        Importante: usamos las categorías del config (no del X) para que la API
        nunca falle si recibe una categoría no vista en training.
        """
        self.le_cat_altitud_ = LabelEncoder()
        self.le_cat_altitud_.fit(self.config_dict["cat_altitud_categorias"])
        return self

    # ────────────────────────────────────────────────────────────────────────
    def transform(self, X):
        """Aplica todo el feature engineering."""
        df = X.copy()

        # Asegurar orden temporal para lags y rolling
        df = df.sort_values(["cod_municipio", "anio", "semana_epi"]).reset_index(drop=True)

        # ── 1. Encode categoría altitud ─────────────────────────────────────
        df["cat_altitud"] = df["cat_altitud"].fillna("Sin dato")
        # Reemplazar valores no vistos por "Sin dato" para evitar errores
        df["cat_altitud"] = df["cat_altitud"].apply(
            lambda x: x if x in self.config_dict["cat_altitud_categorias"] else "Sin dato"
        )
        df["cat_altitud_enc"] = self.le_cat_altitud_.transform(df["cat_altitud"])

        # ── 2. Variables biológicas Aedes aegypti ──────────────────────────
        # Grados-día sobre umbral mínimo de desarrollo (18°C)
        df["grados_dia"] = (df["temp_media_c"] - self.config_dict["temp_min_desarrollo"]).clip(lower=0)

        # Rango óptimo 26-29°C
        df["temp_optima"] = (
            (df["temp_media_c"] >= self.config_dict["temp_optima_min"])
            & (df["temp_media_c"] <= self.config_dict["temp_optima_max"])
        ).astype(int)

        # Temperatura letal >35°C
        df["temp_letal"] = (df["temp_media_c"] > self.config_dict["temp_letal"]).astype(int)

        # Temperatura inhibición <16°C
        df["temp_inhibicion"] = (df["temp_media_c"] < self.config_dict["temp_inhibicion"]).astype(int)

        # ── 3. Índice de idoneidad altitud × temperatura ───────────────────
        # Sigmoide invertida: alta a baja altitud, decae sobre 2200 msnm
        df["indice_idoneidad"] = (
            df["temp_media_c"]
            * (1 / (1 + np.exp(0.003 * (df["altitud_msnm"] - self.config_dict["altitud_inhibicion"]))))
        )

        # ── 4. Precipitación acumulada (rolling por municipio) ─────────────
        df["precip_acum4"] = (
            df.groupby("cod_municipio")["precip_mm"]
            .transform(lambda x: x.rolling(4, min_periods=1).sum())
        )
        df["precip_acum8"] = (
            df.groupby("cod_municipio")["precip_mm"]
            .transform(lambda x: x.rolling(8, min_periods=1).sum())
        )

        # ── 5. Estacionalidad cíclica ──────────────────────────────────────
        df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52)
        df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52)

        # Temporadas de lluvias Colombia (bimodal)
        l1_min, l1_max = self.config_dict["temporada_lluvias_1"]
        l2_min, l2_max = self.config_dict["temporada_lluvias_2"]
        df["temporada_lluvias"] = df["semana_epi"].apply(
            lambda s: 1 if (l1_min <= s <= l1_max) or (l2_min <= s <= l2_max) else 0
        )

        # ── 6. Años epidémicos / post-epidemia ─────────────────────────────
        df["año_epidemico"] = df["anio"].isin(self.config_dict["anos_epidemicos"]).astype(int)
        df["post_epidemia"] = df["anio"].isin(self.config_dict["anos_post_epidemia"]).astype(int)

        # ── 7. Autoregresión de casos ──────────────────────────────────────
        # Solo si la columna `casos` existe (en training sí, en inferencia depende)
        if "casos" in df.columns:
            for lag in [1, 2, 3, 4]:
                df[f"casos_lag{lag}"] = df.groupby("cod_municipio")["casos"].shift(lag)

            df["casos_ma4"] = (
                df.groupby("cod_municipio")["casos"]
                .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
            )

            df["casos_tendencia"] = df["casos_lag1"] - df["casos_lag2"]
        else:
            # Inferencia sin contexto: rellenamos con NaN; se manejará downstream
            for lag in [1, 2, 3, 4]:
                df[f"casos_lag{lag}"] = np.nan
            df["casos_ma4"] = np.nan
            df["casos_tendencia"] = np.nan

        # ── 8. Seleccionar SOLO las features del modelo en el orden correcto ─
        df_out = df[self.config_dict["model_features"]].copy()

        # XGBoost maneja NaN nativamente, no imputamos
        return df_out
