"""
model/processing/data_manager.py
================================
Funciones de I/O del paquete:
  - load_dataset:     carga el CSV de entrenamiento
  - save_pipeline:    persiste el .pkl entrenado
  - load_pipeline:    carga el .pkl al iniciar la API
  - build_canal_endemico:  calcula P25/mediana/P75 por (municipio, semana)
  - apply_canal_endemico:  marca alerta=1 si casos >= P75 (target)
  - get_municipio_context: obtiene histórico de un municipio para construir lags
"""

from pathlib import Path
from typing import List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from model import __version__ as _version
from model.config.core import config, TRAINED_MODEL_DIR, DATASET_DIR


# ── Datasets ─────────────────────────────────────────────────────────────────

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un CSV del directorio de datasets."""
    path = DATASET_DIR / file_name
    df = pd.read_csv(path)

    # cod_municipio siempre como string de 5 chars
    df["cod_municipio"] = df["cod_municipio"].astype(str).str.zfill(5)
    return df


def aggregate_by_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa tipo_dengue sumando casos por (municipio, semana).

    El dataset original tiene varias filas por (municipio, año, semana)
    porque distingue 'grave' vs otros tipos. Para el modelo de alerta
    los sumamos.
    """
    cols_clima = [
        "temp_media_c", "humedad_pct", "precip_mm",
        "temp_lag2", "temp_lag3", "temp_lag4",
        "precip_lag2", "precip_lag3", "precip_lag4",
        "humedad_lag1", "humedad_lag2", "humedad_lag3",
    ]
    keys = ["cod_municipio", "NOM_MPIO", "NOM_DPTO",
            "anio", "semana_epi", "altitud_msnm",
            "cat_altitud", "poblacion"] + cols_clima

    df_mun = (
        df.groupby(keys)
        .agg(casos=("casos", "sum"))
        .reset_index()
    )
    df_mun["tasa_x100k"] = (df_mun["casos"] / df_mun["poblacion"]) * 100000
    return df_mun


def get_municipios_validos(df_mun: pd.DataFrame) -> List[str]:
    """
    Municipios con >= min_semanas_con_casos en período de entrenamiento.
    Los municipios con pocas semanas no permiten calcular percentiles confiables.
    """
    df_train = df_mun[df_mun["anio"].isin(config.model_config_.train_years)].copy()
    semanas_por_mun = (
        df_train[df_train["casos"] > 0]
        .groupby("cod_municipio")
        .size()
    )
    return semanas_por_mun[
        semanas_por_mun >= config.model_config_.min_semanas_con_casos
    ].index.tolist()


def build_grid_completa(df_mun: pd.DataFrame, municipios_validos: List[str]) -> pd.DataFrame:
    """
    Construye grilla completa: municipios x años x semanas (1-52).
    Llena con casos=0 las semanas sin registro.
    """
    cols_clima = [
        "temp_media_c", "humedad_pct", "precip_mm",
        "temp_lag2", "temp_lag3", "temp_lag4",
        "precip_lag2", "precip_lag3", "precip_lag4",
        "humedad_lag1", "humedad_lag2", "humedad_lag3",
    ]
    todos_anos = (
        config.model_config_.train_years
        + config.model_config_.validation_years
        + config.model_config_.test_years
    )

    grilla = pd.MultiIndex.from_product(
        [municipios_validos, todos_anos, range(1, 53)],
        names=["cod_municipio", "anio", "semana_epi"],
    ).to_frame(index=False)

    # Info fija por municipio
    info_mun = (
        df_mun[df_mun["cod_municipio"].isin(municipios_validos)]
        [["cod_municipio", "NOM_MPIO", "NOM_DPTO", "altitud_msnm", "cat_altitud"]]
        .drop_duplicates("cod_municipio")
    )

    casos_ref = (
        df_mun[df_mun["cod_municipio"].isin(municipios_validos)]
        [["cod_municipio", "anio", "semana_epi", "casos", "tasa_x100k"]]
        .drop_duplicates(["cod_municipio", "anio", "semana_epi"])
    )

    clima_ref = (
        df_mun[df_mun["cod_municipio"].isin(municipios_validos)]
        [["cod_municipio", "anio", "semana_epi"] + cols_clima]
        .drop_duplicates(["cod_municipio", "anio", "semana_epi"])
    )

    pob_ref = (
        df_mun[df_mun["cod_municipio"].isin(municipios_validos)]
        .groupby(["cod_municipio", "anio"])["poblacion"]
        .first()
        .reset_index()
    )

    df_completo = (
        grilla
        .merge(casos_ref, on=["cod_municipio", "anio", "semana_epi"], how="left")
        .merge(pob_ref, on=["cod_municipio", "anio"], how="left")
        .merge(clima_ref, on=["cod_municipio", "anio", "semana_epi"], how="left")
        .merge(info_mun, on="cod_municipio", how="left")
    )

    df_completo["casos"] = df_completo["casos"].fillna(0)
    df_completo["tasa_x100k"] = df_completo["tasa_x100k"].fillna(0)

    # Interpolar clima dentro de cada municipio
    df_completo = df_completo.sort_values(["cod_municipio", "anio", "semana_epi"])
    for col in cols_clima:
        df_completo[col] = (
            df_completo.groupby("cod_municipio")[col]
            .transform(lambda x: x.interpolate().ffill().bfill())
        )

    return df_completo


# ── Canal endémico ───────────────────────────────────────────────────────────

def build_canal_endemico(df_completo: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula P25 / mediana / P75 de casos por (municipio, semana)
    usando SOLO años de entrenamiento y SOLO semanas con casos > 0.
    """
    canal_years = config.model_config_.canal_endemico_years
    df_train = df_completo[df_completo["anio"].isin(canal_years)]
    df_with_cases = df_train[df_train["casos"] > 0].copy()

    canal_mun = (
        df_with_cases.groupby(["cod_municipio", "semana_epi"])["casos"]
        .agg(
            p25=lambda x: x.quantile(0.25),
            mediana=lambda x: x.quantile(0.50),
            p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    return canal_mun


def apply_canal_endemico(df_completo: pd.DataFrame, canal_mun: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza el canal endémico con el dataset y calcula:
      - zona_canal: epidemica / alerta / exito / seguridad
      - alerta: target binario (epidemica o alerta → 1)
    """
    df = df_completo.merge(canal_mun, on=["cod_municipio", "semana_epi"], how="left")
    df["p25"] = df["p25"].fillna(0)
    df["mediana"] = df["mediana"].fillna(0)
    df["p75"] = df["p75"].fillna(0)

    def clasificar(row):
        if pd.isna(row["p25"]):
            return "seguridad"
        if row["p75"] == 0 and row["casos"] == 0:
            return "seguridad"
        if row["p25"] == row["p75"]:
            if row["casos"] > row["p75"]:
                return "epidemica"
            elif row["casos"] == row["p75"] and row["casos"] > 0:
                return "alerta"
            elif row["casos"] == 0:
                return "seguridad"
            else:
                return "exito"
        if row["casos"] >= row["p75"]:
            return "epidemica"
        elif row["casos"] >= row["mediana"]:
            return "alerta"
        elif row["casos"] >= row["p25"]:
            return "exito"
        else:
            return "seguridad"

    df["zona_canal"] = df.apply(clasificar, axis=1)
    df["alerta"] = df["zona_canal"].map(
        {"epidemica": 1, "alerta": 1, "exito": 0, "seguridad": 0}
    )
    return df


# ── Persistencia del pipeline ────────────────────────────────────────────────

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persiste el pipeline entrenado como .pkl en trained/."""
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    # Limpiar versiones anteriores
    remove_old_pipelines(files_to_keep=[save_file_name])

    joblib.dump(pipeline_to_persist, save_path)
    print(f"✓ Pipeline guardado en: {save_path}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Carga el pipeline persistido."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """Borra pipelines viejos para evitar acumulación."""
    do_not_delete = files_to_keep + ["__init__.py", ".gitkeep"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


# ── Contexto histórico para inferencia ───────────────────────────────────────

def get_municipio_context(cod_municipio: str, anio: int, semana_epi: int,
                          n_weeks_back: int = 8) -> pd.DataFrame:
    """
    Para inferencia: lee del CSV histórico las últimas N semanas
    del municipio para poder construir lags.

    Returns
    -------
    DataFrame con las últimas N semanas anteriores a (anio, semana_epi).
    """
    df = load_dataset(file_name=config.app_config.train_data_file)
    df_mun = aggregate_by_municipio(df)

    # Filtrar histórico anterior a la semana objetivo
    df_hist = df_mun[df_mun["cod_municipio"] == cod_municipio].copy()
    df_hist = df_hist.sort_values(["anio", "semana_epi"])

    # Tomar las últimas N semanas anteriores
    mask = (df_hist["anio"] < anio) | (
        (df_hist["anio"] == anio) & (df_hist["semana_epi"] < semana_epi)
    )
    df_hist = df_hist[mask].tail(n_weeks_back).copy()
    return df_hist
