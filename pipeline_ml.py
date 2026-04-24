import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ================================
# 1. Configuracion Carpetas
# ================================
DATA_PATH = "data/processed/dengue_full_final.csv"
OUTPUT_FOLDER = "outputs/models/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEPARTAMENTOS = ["ARAUCA", "CASANARE", "META"]

# ================================
# 2. Cargar Data
# ================================
df = pd.read_csv(DATA_PATH)

print("Dataset original:", df.shape)

df["departamento_ocurrencia"] = (
    df["departamento_ocurrencia"]
    .astype(str).str.upper().str.strip()
)

df = df[df["departamento_ocurrencia"].isin(DEPARTAMENTOS)]

print("Dataset filtrado:", df.shape)

# ================================
# 3. Feature
# ================================
df = df.sort_values(["cod_mun_o", "ano", "semana"])

df["casos"] = df["casos"].replace(0, 1)
df["poblacion"] = df["poblacion"].replace(0, 1)

df["incidencia"] = (df["casos"] / df["poblacion"]) * 100000

# crecimiento población
df["pob_lag1"] = df.groupby("cod_mun_o")["poblacion"].shift(1)
df["crecimiento_pob"] = (df["poblacion"] - df["pob_lag1"]) / df["pob_lag1"]

# per capita
df["casos_per_capita"] = df["casos"] / df["poblacion"]

# clima
df["precip_lag1"] = df.groupby("cod_mun_o")["precipitation"].shift(1)
df["precip_lag2"] = df.groupby("cod_mun_o")["precipitation"].shift(2)

df["precip_rolling_3"] = (
    df.groupby("cod_mun_o")["precipitation"]
    .rolling(3).mean().reset_index(level=0, drop=True)
)

# interacciones
df["clima_poblacion"] = df["precipitation"] * df["poblacion"]
df["lag1_poblacion"] = df["precip_lag1"] * df["poblacion"]

# dengue lags
df["casos_lag1"] = df.groupby("cod_mun_o")["casos"].shift(1)
df["casos_lag2"] = df.groupby("cod_mun_o")["casos"].shift(2)
df["casos_lag3"] = df.groupby("cod_mun_o")["casos"].shift(3)

# ================================
# 4. Target
# ================================
percentil = 0.8

df["alerta"] = df.groupby("departamento_ocurrencia")["casos"]\
    .transform(lambda x: x > x.quantile(percentil)).astype(int)

print("Distribución target:")
print(df["alerta"].value_counts(normalize=True))

# ================================
# 5. Features
# ================================
features = [
    "precipitation","precip_lag1","precip_lag2","precip_rolling_3",
    "poblacion","crecimiento_pob","casos_per_capita",
    "clima_poblacion","lag1_poblacion",
    "casos_lag1","casos_lag2","casos_lag3"
]

target = "alerta"

# ================================
# 6. Limpieza
# ================================
df = df.dropna(subset=["casos"])

df = df.groupby("cod_mun_o").apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
df = df.fillna(0)

print("Dataset final:", df.shape)

# ================================
# 7. Split
# ================================
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ================================
# 8. Clasificación
# ================================
positivos = y_train.sum()
negativos = len(y_train) - positivos

scale_pos_weight = negativos / positivos if positivos > 0 else 1

print(f"Positivos: {positivos} | Negativos: {negativos}")

# ================================
# 7. Estandarización
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 8. Modelos
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
}

# ================================
# 9. Evaluación - Metricas
# ================================
def evaluate_model(y_true, y_prob, threshold=0.3):

    y_pred = (y_prob > threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return report, roc, cm

results = []

# ================================
# 10. Entrenamiento
# ================================
for name, model in models.items():

    print(f"\nEntrenando {name}")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:,1]

    report, roc, cm = evaluate_model(y_test, y_prob, threshold=0.3)

    print("Matriz de confusión:\n", cm)

    results.append({
        "modelo": name,
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc": roc
    })

# ================================
# 10. Resultados
# ================================
df_results = pd.DataFrame(results).sort_values(by="recall", ascending=False)

print("\nRESULTADOS FINALES:")
print(df_results)

df_results.to_csv(OUTPUT_FOLDER + "model_results.csv", index=False)