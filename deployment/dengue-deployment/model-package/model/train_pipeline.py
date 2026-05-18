"""
model/train_pipeline.py
=======================
Entrena el pipeline completo y persiste el .pkl.

Ejecutar:
    python model/train_pipeline.py

O via tox:
    tox run -e train

Flujo:
  1. Carga dataset crudo
  2. Agrega por municipio (colapsa tipo_dengue)
  3. Filtra municipios válidos (>= 20 semanas con casos en train)
  4. Construye grilla completa (incluye semanas con cero casos)
  5. Calcula canal endémico (P25/mediana/P75 solo con train)
  6. Genera target binario alerta
  7. Entrena Pipeline(FeatureEngineer + XGBoost) con train (2009-2016)
  8. Evalúa en validation (2017-2019) y test (2022-2024)
  9. Persiste .pkl
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)

from model.config.core import config
from model.pipeline import dengue_pipeline
from model.processing.data_manager import (
    load_dataset,
    aggregate_by_municipio,
    get_municipios_validos,
    build_grid_completa,
    build_canal_endemico,
    apply_canal_endemico,
    save_pipeline,
)


def run_training() -> None:
    """Pipeline completo de entrenamiento."""

    print("=" * 60)
    print("ENTRENAMIENTO — AI-lerta Dengue Colombia")
    print("=" * 60)

    # ── 1. Carga ────────────────────────────────────────────────────────────
    print("\n[1/9] Cargando dataset...")
    df = load_dataset(file_name=config.app_config.train_data_file)
    print(f"      {len(df):,} filas, {df['cod_municipio'].nunique()} municipios")

    # ── 2. Agregar por municipio ────────────────────────────────────────────
    print("\n[2/9] Agregando tipo_dengue por municipio...")
    df_mun = aggregate_by_municipio(df)
    print(f"      {len(df_mun):,} filas tras colapsar tipo_dengue")

    # ── 3. Municipios válidos ───────────────────────────────────────────────
    print("\n[3/9] Filtrando municipios válidos...")
    municipios_validos = get_municipios_validos(df_mun)
    print(f"      {len(municipios_validos)} municipios con >= "
          f"{config.model_config_.min_semanas_con_casos} semanas con casos en train")

    # ── 4. Grilla completa ──────────────────────────────────────────────────
    print("\n[4/9] Construyendo grilla completa (incluye ceros)...")
    df_completo = build_grid_completa(df_mun, municipios_validos)
    print(f"      {len(df_completo):,} filas en grilla")

    # ── 5. Canal endémico ───────────────────────────────────────────────────
    print("\n[5/9] Calculando canal endémico (P25/mediana/P75)...")
    canal_mun = build_canal_endemico(df_completo)
    print(f"      {len(canal_mun):,} filas de canal endémico")

    # ── 6. Aplicar canal → target ──────────────────────────────────────────
    print("\n[6/9] Generando target binario alerta...")
    df_modelo = apply_canal_endemico(df_completo, canal_mun)
    pct_alerta = df_modelo["alerta"].mean() * 100
    print(f"      Distribución zonas:")
    print(df_modelo["zona_canal"].value_counts().to_string())
    print(f"      % alerta global: {pct_alerta:.1f}%")

    # ── 7. Split train / validation / test ──────────────────────────────────
    print("\n[7/9] Split temporal...")

    df_tr = df_modelo[df_modelo["anio"].isin(config.model_config_.train_years)].copy()
    df_val = df_modelo[df_modelo["anio"].isin(config.model_config_.validation_years)].copy()
    df_te = df_modelo[df_modelo["anio"].isin(config.model_config_.test_years)].copy()

    # El Pipeline necesita X con TODAS las columnas crudas (no las features finales)
    # porque el FeatureEngineer genera las features dentro
    X_tr = df_tr.drop(columns=["alerta", "zona_canal", "p25", "mediana", "p75", "tasa_x100k"])
    y_tr = df_tr["alerta"].astype(int)

    X_val = df_val.drop(columns=["alerta", "zona_canal", "p25", "mediana", "p75", "tasa_x100k"])
    y_val = df_val["alerta"].astype(int)

    X_te = df_te.drop(columns=["alerta", "zona_canal", "p25", "mediana", "p75", "tasa_x100k"])
    y_te = df_te["alerta"].astype(int)

    print(f"      TRAIN      : {len(X_tr):,} | alerta: {y_tr.mean()*100:.1f}%")
    print(f"      VALIDATION : {len(X_val):,} | alerta: {y_val.mean()*100:.1f}%")
    print(f"      TEST       : {len(X_te):,} | alerta: {y_te.mean()*100:.1f}%")

    # ── 8. Entrenar Pipeline ────────────────────────────────────────────────
    print("\n[8/9] Entrenando Pipeline (FeatureEngineer + XGBoost)...")
    dengue_pipeline.fit(X_tr, y_tr)
    print("      ✓ Pipeline entrenado")

    # Buscar umbral óptimo en validation
    print("\n      Buscando umbral óptimo (max F2 en validation)...")
    y_prob_val = dengue_pipeline.predict_proba(X_val)[:, 1]
    mejor_f2, mejor_u = 0, 0.5
    for u in np.arange(0.1, 0.9, 0.05):
        yp = (y_prob_val >= u).astype(int)
        if yp.sum() == 0 or (1 - yp).sum() == 0:
            continue
        f2 = fbeta_score(y_val, yp, beta=2, zero_division=0)
        if f2 > mejor_f2:
            mejor_f2, mejor_u = f2, round(u, 2)
    print(f"      Umbral óptimo: {mejor_u} | F2 en val: {mejor_f2:.3f}")

    # Evaluación final
    for nombre, X, y in [
        ("VALIDATION", X_val, y_val),
        ("TEST      ", X_te, y_te),
    ]:
        y_prob = dengue_pipeline.predict_proba(X)[:, 1]
        y_pred = (y_prob >= mejor_u).astype(int)
        print(f"\n      === {nombre} ===")
        print(f"      F2     : {fbeta_score(y, y_pred, beta=2):.3f}")
        print(f"      Recall : {recall_score(y, y_pred):.3f}")
        print(f"      Prec   : {precision_score(y, y_pred):.3f}")
        print(f"      AUC    : {roc_auc_score(y, y_prob):.3f}")

    # ── 9. Persistir ────────────────────────────────────────────────────────
    print("\n[9/9] Persistiendo pipeline entrenado...")
    save_pipeline(pipeline_to_persist=dengue_pipeline)
    print("=" * 60)
    print("✓ Entrenamiento completado")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
