import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_PATH = "data/processed/dengue_full_final.csv"
OUTPUT_FOLDER = "outputs/series_temporales/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEPARTAMENTOS = ["ARAUCA", "CASANARE", "META"]

# ==========================================
# CARGAR DATASET
# ==========================================
df = pd.read_csv(DATA_PATH)

print("Dataset cargado:", df.shape)

# ==========================================
# LIMPIAR NOMBRES
# ==========================================
df["departamento_ocurrencia"] = (
    df["departamento_ocurrencia"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ==========================================
# CREAR FECHA
# ==========================================
df["fecha"] = pd.to_datetime(
    df["ano"].astype(str) + "-W" + df["semana"].astype(str) + "-1",
    format="%G-W%V-%u"
)

# ==========================================
# FUNCIÓN NORMALIZAR
# ==========================================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# ==========================================
# GRAFICAR CADA DEPARTAMENTO
# ==========================================
for depto in DEPARTAMENTOS:

    print(f"Procesando {depto}...")

    df_temp = df[df["departamento_ocurrencia"] == depto].copy()

    # Agrupar por fecha semanal
    df_temp = df_temp.groupby("fecha").agg({
        "casos": "sum",
        "precipitation": "mean"
    }).reset_index()

    # Normalizar para comparar curvas
    df_temp["casos_norm"] = normalize(df_temp["casos"])
    df_temp["precip_norm"] = normalize(df_temp["precipitation"])

    # ==========================================
    # GRAFICA PRINCIPAL
    # ==========================================
    plt.figure(figsize=(15,6))

    plt.plot(
        df_temp["fecha"],
        df_temp["casos_norm"],
        label="Casos dengue",
        linewidth=2
    )

    plt.plot(
        df_temp["fecha"],
        df_temp["precip_norm"],
        label="Precipitación",
        linewidth=2
    )

    plt.title(f"Serie temporal Dengue vs Precipitación - {depto}", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Valor normalizado (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FOLDER}{depto}_serie_temporal.png", dpi=300)
    plt.close()

print("\n✅ Gráficas generadas en:", OUTPUT_FOLDER)