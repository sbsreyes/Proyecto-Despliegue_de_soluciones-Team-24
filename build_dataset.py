import pandas as pd
import glob
import os
import ee
import time

# ================================
# 1. Configuracion Carpetas
# ================================
RAW_PATH = "data/raw"
DEMOGRAPHIC_PATH = "data/demographic"
OUTPUT_PATH = "data/processed/dengue_full_final.csv"

PROJECT_ID = "dengue-ml-project"

YEARS = [2016, 2017, 2018, 2019, 2023, 2024]
MONTHS = list(range(1, 13))

# ================================
# 2. Init
# ================================
def init_gee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

# ================================
# 3. Cargar SIVIGILA
# ================================
def load_and_unify_data(path_folder):
    files = glob.glob(os.path.join(path_folder, "*.xlsx"))
    df_list = []

    for file in files:
        print(f"Cargando: {file}")
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.lower()
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    return df

# ================================
# 4. Filtros
# ================================
def select_and_filter(df):

    cols = [
        "ano","semana","cod_mun_o","municipio_ocurrencia",
        "departamento_ocurrencia","nombre_evento",
        "nom_est_f_caso","edad","sexo","area"
    ]

    df = df[cols]
    df = df[df["nombre_evento"].str.upper() == "DENGUE"]

    df["confirmado"] = (
        df["nom_est_f_caso"]
        .astype(str)
        .str.upper()
        .str.contains("CONFIRMADO")
        .astype(int)
    )

    return df

# ================================
# 5. Columnas agregadas
# ================================
def aggregate_weekly(df):

    df = df.groupby(
        ["cod_mun_o","municipio_ocurrencia",
         "departamento_ocurrencia","ano","semana"]
    ).agg(
        casos=("nombre_evento","count"),
        confirmados=("confirmado","sum"),
        edad_promedio=("edad","mean")
    ).reset_index()

    return df

# ================================
# 6. Cargar DANE
# ================================
def load_dane(path_folder):

    files = glob.glob(os.path.join(path_folder, "*.xlsx"))
    df_list = []

    for file in files:
        df = pd.read_excel(file)
        df.columns = df.columns.str.lower().str.strip()

        df = df[df["area geografica"].str.upper() == "TOTAL"]

        df = df.rename(columns={
            "mpio": "cod_mpio",
            "año": "year",
            "población": "poblacion"
        })

        df["poblacion"] = (
            df["poblacion"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .astype(float)
        )

        df["cod_mpio"] = df["cod_mpio"].astype(str).str.zfill(5)
        df["year"] = df["year"].astype(int)

        df_list.append(df[["cod_mpio","year","poblacion"]])

    df_demo = pd.concat(df_list, ignore_index=True)
    df_demo = df_demo.drop_duplicates(["cod_mpio","year"])

    return df_demo

# ================================
# 7. MERGE DANE
# ================================
def merge_dane(df, df_demo):

    df["cod_mun_o"] = df["cod_mun_o"].astype(str).str.zfill(5)

    return df.merge(
        df_demo,
        left_on=["cod_mun_o","ano"],
        right_on=["cod_mpio","year"],
        how="left"
    )

# ================================
# 8. Clima
# ================================
def extract_climate_chunk(year, month):

    region = ee.Geometry.Rectangle([-79, -4, -66, 13])

    start = f"{year}-{str(month).zfill(2)}-01"
    end = f"{year}-{str(month+1).zfill(2)}-01" if month < 12 else f"{year}-12-31"

    # Precipitación
    rain = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(region)
    )

    # Temperatura + humedad (ERA5)
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .filterBounds(region)
    )

    def map_func(img):

        date = img.date().format("YYYY-MM-dd")

        rain_val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000,
            maxPixels=1e13
        ).get("precipitation")

        era_img = era5.filterDate(img.date(), img.date().advance(1, 'day')).first()

        temp = era_img.select("temperature_2m").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000
        ).get("temperature_2m")

        humidity = era_img.select("dewpoint_temperature_2m").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000
        ).get("dewpoint_temperature_2m")

        return ee.Feature(None, {
            "date": date,
            "precipitation": rain_val,
            "temperature": temp,
            "humidity": humidity
        })

    fc = rain.map(map_func)

    try:
        data = fc.getInfo()["features"]
    except:
        return None

    return pd.DataFrame([f["properties"] for f in data])

# ================================
# 9. Clima
# ================================
def build_climate_dataset():

    df_list = []

    for year in YEARS:
        for month in MONTHS:

            print(f"Clima {year}-{month}")
            df_chunk = extract_climate_chunk(year, month)

            if df_chunk is not None and not df_chunk.empty:
                df_list.append(df_chunk)

            time.sleep(1)

    df = pd.concat(df_list, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["ano"] = df["date"].dt.year
    df["semana"] = df["date"].dt.isocalendar().week

    df = df.groupby(["ano","semana"]).mean(numeric_only=True).reset_index()

    return df

# ================================
# 10. Clima
# ================================
def merge_climate(df, df_clima):
    return df.merge(df_clima, on=["ano","semana"], how="left")

# ================================
# 11. Features
# ================================
def create_features(df):

    df["casos"] = df["casos"].replace(0, 1)

    df["incidencia"] = (df["casos"] / df["poblacion"]) * 100000
    df["ratio_confirmados"] = df["confirmados"] / df["casos"]

    df = df.sort_values(["cod_mun_o","ano","semana"])

    # LAGS casos
    df["lag_1"] = df.groupby("cod_mun_o")["casos"].shift(1)
    df["lag_2"] = df.groupby("cod_mun_o")["casos"].shift(2)
    df["lag_3"] = df.groupby("cod_mun_o")["casos"].shift(3)

    # 🔥 LAGS clima
    df["precip_lag1"] = df.groupby("cod_mun_o")["precipitation"].shift(1)
    df["precip_lag2"] = df.groupby("cod_mun_o")["precipitation"].shift(2)

    df["temp_lag1"] = df.groupby("cod_mun_o")["temperature"].shift(1)
    df["hum_lag1"] = df.groupby("cod_mun_o")["humidity"].shift(1)

    # ALERTA
    threshold = df["casos"].mean() + 2 * df["casos"].std()
    df["alerta"] = (df["casos"] > threshold).astype(int)

    return df

# ================================
# 12. Guardar
# ================================
def save(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Dataset final listo")

# ================================
# 13. MAIN
# ================================
def main():

    init_gee()

    print("SIVIGILA...")
    df = load_and_unify_data(RAW_PATH)
    df = select_and_filter(df)
    df = aggregate_weekly(df)

    print("DANE...")
    df_demo = load_dane(DEMOGRAPHIC_PATH)
    df = merge_dane(df, df_demo)

    print("CLIMA...")
    df_clima = build_climate_dataset()

    df = merge_climate(df, df_clima)

    print("FEATURE ENGINEERING...")
    df = create_features(df)

    save(df)

if __name__ == "__main__":
    main()