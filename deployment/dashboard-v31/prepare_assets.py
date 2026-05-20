"""
prepare_assets.py
=================
Convierte divipola.parquet a divipola.csv en assets/ para el dashboard.

Uso:
    # Asegúrate de tener divipola.parquet en la carpeta del proyecto
    python prepare_assets.py
"""
import os
import pandas as pd
from pathlib import Path

# Crear carpeta assets si no existe
Path("assets").mkdir(exist_ok=True)

# Cargar parquet
print("Cargando divipola.parquet...")
df = pd.read_parquet("divipola.parquet")

print(f"Shape: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

# Validar que tiene las columnas necesarias
required = ["cod_municipio", "nom_municipio", "cod_departamento",
            "nom_departamento", "latitud", "longitud"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"⚠️  Columnas faltantes: {missing}")
    print(f"   Columnas disponibles: {df.columns.tolist()}")
    raise SystemExit(1)

# Solo las columnas necesarias
df_out = df[required].copy()
df_out["cod_municipio"] = df_out["cod_municipio"].astype(str).str.zfill(5)
df_out["cod_departamento"] = df_out["cod_departamento"].astype(str).str.zfill(2)

# Guardar
output_path = "assets/divipola.csv"
df_out.to_csv(output_path, index=False)

print(f"\n✓ Guardado: {output_path}")
print(f"  {len(df_out)} municipios, {df_out['nom_departamento'].nunique()} departamentos")
print(f"\nMuestra:")
print(df_out.head())
