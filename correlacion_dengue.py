import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================
# 1. Configuracion Carpetas
# ================================
DATA_PATH = "data/processed/dengue_full_final.csv"
OUTPUT_FOLDER = "outputs/series_temporales/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEPARTAMENTOS = ["ARAUCA", "CASANARE", "META"]

# ================================
# 2. Cargar Data
# ================================
df = pd.read_csv(DATA_PATH)

print("Dataset cargado:", df.shape)

# ================================
# 3. Crear Fecha
# ================================
# Convertir año + semana a fecha real
df["fecha"] = pd.to_datetime(
    df["ano"].astype(str) + "-W" + df["semana"].astype(str) + "-1",
    format="%G-W%V-%u"
)

df = df.sort_values("fecha")

# ================================
# 4. Normalizar Variables
# ================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# ================================
# 5. Series por Departamento
# ================================
for depto in DEPARTAMENTOS:

    print(f"Procesando {depto}...")

    df_temp = df[df["departamento_ocurrencia"] == depto].copy()

    if df_temp.empty:
        print(f"⚠️ No hay datos para {depto}")
        continue

    # Agrupar por semana (por si hay duplicados)
    df_temp = df_temp.groupby("fecha").agg({
        "casos": "sum",
        "precipitation": "mean"
    }).reset_index()

    # ================================
    # Normalizar (Visualización)
    # ================================
    df_temp["casos_norm"] = normalize(df_temp["casos"])
    df_temp["precip_norm"] = normalize(df_temp["precipitation"])

    # ================================
    # Gráfica
    # ================================
    plt.figure(figsize=(14,6))

    plt.plot(df_temp["fecha"], df_temp["casos_norm"], label="Casos (normalizado)")
    plt.plot(df_temp["fecha"], df_temp["precip_norm"], label="Precipitación (normalizado)")

    plt.title(f"Serie temporal dengue vs clima - {depto}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor normalizado")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FOLDER}serie_{depto}.png")
    plt.close()

    # ================================
    # Gráfica con Lags
    # ================================
    df_temp["precip_lag1"] = df_temp["precip_norm"].shift(1)
    df_temp["precip_lag2"] = df_temp["precip_norm"].shift(2)

    plt.figure(figsize=(14,6))

    plt.plot(df_temp["fecha"], df_temp["casos_norm"], label="Casos")
    plt.plot(df_temp["fecha"], df_temp["precip_lag1"], label="Precipitación lag 1")
    plt.plot(df_temp["fecha"], df_temp["precip_lag2"], label="Precipitación lag 2")

    plt.title(f"Efecto retardado del clima - {depto}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor normalizado")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FOLDER}lags_{depto}.png")
    plt.close()

    # ================================
    # Gráfica por año
    # ================================
    df_temp["year"] = df_temp["fecha"].dt.year

    for year in sorted(df_temp["year"].unique()):

        df_year = df_temp[df_temp["year"] == year]

        plt.figure(figsize=(12,5))

        plt.plot(df_year["fecha"], df_year["casos_norm"], label="Casos")
        plt.plot(df_year["fecha"], df_year["precip_norm"], label="Precipitación")

        plt.title(f"{depto} - Año {year}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor normalizado")
        plt.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"{OUTPUT_FOLDER}{depto}_{year}.png")
        plt.close()

print("\n✅ Series temporales generadas en:", OUTPUT_FOLDER)