# ==========================================================================
# AI-lerta: sistema de alerta temprana para dengue en municipios de Colombia
# mediante canal endémico, variables climáticas y aprendizaje automático
#
# Dashboard interactivo — consume la API REST desplegada en AWS EC2
#
# Proyecto Maestría IA
# Universidad de los Andes
# Autores: Juan Sebastián Reyes Acosta, Julián Felipe Moncada Castro,
#          Irlanda Katiuzhca Robles Orellana, Amalia Catalina Lehmann Oliveros
# ==========================================================================
#
# ARQUITECTURA:
#   [Usuario] → [Dashboard Dash] → [API REST EC2] → [Modelo XGBoost]
#
# El dashboard:
#   • NO carga el modelo localmente
#   • Hace VLOOKUP de clima 2025 por departamento para pre-llenar campos
#   • Permite al usuario modificar todos los inputs (clima + casos)
#   • Envía al endpoint /predict/realtime
#   • Clasifica la probabilidad retornada en DOS escalas complementarias:
#       - Zona INS (Epidémica/Alerta/Éxito/Seguridad) — convención epidemiológica
#       - Nivel de Urgencia (ALTA/MEDIA/PRECAUCIÓN/NORMAL) — escala operacional
# ==========================================================================

import os
import unicodedata
import requests
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://54.226.141.151:8001")
PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict/realtime"
HEALTH_ENDPOINT = f"{API_URL}/api/v1/health"
REQUEST_TIMEOUT = 15

DEFAULT_ANIO = 2025


# ──────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────

def normalize_depto(s: str) -> str:
    """
    Normaliza nombre de departamento para hacer matching entre DIVIPOLA
    (mayúsculas con tilde) y clima_colombia_2025 (mixto sin tilde).
    """
    if not isinstance(s, str):
        return ""
    # quitar tildes
    nfkd = unicodedata.normalize("NFKD", s)
    no_tildes = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return no_tildes.upper().strip()


# ──────────────────────────────────────────────────────────────────────────
# CARGA DE CATÁLOGOS
# ──────────────────────────────────────────────────────────────────────────

def load_divipola() -> pd.DataFrame:
    """Carga el catálogo DIVIPOLA con municipios y coordenadas."""
    path = "assets/divipola.csv"
    try:
        df = pd.read_csv(path, dtype={"cod_municipio": str, "cod_departamento": str})
        df["cod_municipio"] = df["cod_municipio"].str.zfill(5)
        df["cod_departamento"] = df["cod_departamento"].str.zfill(2)
        df["nom_departamento_norm"] = df["nom_departamento"].apply(normalize_depto)
        return df.sort_values(["nom_departamento", "nom_municipio"]).reset_index(drop=True)
    except FileNotFoundError:
        print(f"⚠️ No se encontró {path}. Ejecuta prepare_assets.py primero.")
        return pd.DataFrame()


def load_clima_2025() -> pd.DataFrame:
    """Carga el catálogo de clima 2025 por departamento × semana."""
    path = "assets/clima_colombia_2025.csv"
    try:
        df = pd.read_csv(path)
        df["depto_norm"] = df["departamento"].apply(normalize_depto)
        return df
    except FileNotFoundError:
        print(f"⚠️ No se encontró {path}. Pre-rellenado de clima no funcionará.")
        return pd.DataFrame()


df_divipola = load_divipola()
df_clima = load_clima_2025()

DEPARTAMENTOS = sorted(df_divipola["nom_departamento"].unique()) if not df_divipola.empty else []

print(f"✓ DIVIPOLA: {len(df_divipola)} municipios, {len(DEPARTAMENTOS)} departamentos")
print(f"✓ Clima 2025: {len(df_clima)} registros")


# ──────────────────────────────────────────────────────────────────────────
# CLIENTES DE LA API
# ──────────────────────────────────────────────────────────────────────────

def predict_via_api(payload: dict) -> dict:
    """Llama POST /api/v1/predict/realtime con el payload."""
    try:
        r = requests.post(PREDICT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": f"Timeout: la API no respondió en {REQUEST_TIMEOUT}s"}
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar a la API en {API_URL}"}
    except requests.exceptions.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": f"HTTP {r.status_code}: {detail}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


def check_api_health() -> dict:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.json() if r.ok else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────
# CLASIFICADORES (la lógica de niveles vive en el dashboard)
# ──────────────────────────────────────────────────────────────────────────

def clasificar_nivel_urgencia(prob: float) -> dict:
    """
    Nivel de Urgencia: escala operacional para tomadores de decisión.
    Basado en umbrales del clasificador binario.
    """
    if prob >= 0.75:
        return {"nivel": "ALTA", "emoji": "🔴", "color": "danger",
                "desc": "Acción inmediata requerida"}
    elif prob >= 0.50:
        return {"nivel": "MEDIA", "emoji": "🟠", "color": "warning",
                "desc": "Vigilancia activa reforzada"}
    elif prob >= 0.30:
        return {"nivel": "PRECAUCIÓN", "emoji": "🟡", "color": "info",
                "desc": "Monitoreo aumentado"}
    else:
        return {"nivel": "NORMAL", "emoji": "🟢", "color": "success",
                "desc": "Vigilancia rutinaria"}


def clasificar_zona_ins(prob: float, casos_actual: float) -> dict:
    """
    Zona del INS: convención epidemiológica colombiana.

    Mapea la probabilidad del modelo a las 4 zonas del canal endémico:
      - Epidémica:  prob alta sostenida (≥ P75 histórico)
      - Alerta:     prob moderada-alta (entre P50 y P75)
      - Éxito:      prob baja-moderada (entre P25 y P50)
      - Seguridad:  prob baja (< P25 o sin casos)

    Caso especial INS: P75=0 y casos=0 → Seguridad.
    """
    # Caso especial: sin casos y probabilidad muy baja
    if prob < 0.10 and casos_actual == 0:
        return {"zona": "Seguridad", "color": "success", "emoji": "🟢"}

    # Zonas basadas en distribución empírica de probabilidades del modelo
    # (calibrados para reflejar la prevalencia histórica del INS:
    #  Seguridad 79.9%, Éxito 2.6%, Alerta 9.0%, Epidémica 8.4%)
    if prob >= 0.85:
        return {"zona": "Epidémica", "color": "danger", "emoji": "🔴"}
    elif prob >= 0.50:
        return {"zona": "Alerta", "color": "warning", "emoji": "🟠"}
    elif prob >= 0.30:
        return {"zona": "Éxito", "color": "info", "emoji": "🔵"}
    else:
        return {"zona": "Seguridad", "color": "success", "emoji": "🟢"}


# Healthcheck al arrancar
_health = check_api_health()
if "error" in _health:
    print(f"⚠️  API no disponible: {_health['error']}")
else:
    print(f"✓ API conectada: modelo v{_health.get('model_version', '?')}")


# ──────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "AI-lerta Dengue Colombia"
server = app.server


def crear_tarjeta_kpi(titulo, valor, color_borde="primary", id_dinamico=None):
    return dbc.Card([
        dbc.CardBody([
            html.H6(titulo,
                    className="text-muted text-uppercase font-weight-bold mb-1",
                    style={"fontSize": "11px"}),
            html.H3(valor, id=id_dinamico,
                    className="font-weight-bold m-0",
                    style={"color": "#2C3E50", "fontSize": "22px"})
        ], className="p-3")
    ], style={
        "borderTop": f"4px solid var(--bs-{color_borde})",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
        "height": "100%",
        "borderRadius": "8px"
    })


# ──────────────────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ─── HEADER ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🦟 AI-LERTA DENGUE COLOMBIA",
                        style={"textAlign": "center", "fontWeight": "bold",
                               "color": "#00E5FF", "margin": "0"}),
                html.H5("Sistema Inteligente de Alertas Tempranas",
                        style={"textAlign": "center", "color": "lightgray",
                               "margin": "5px 0 0 0", "fontSize": "16px"}),
                html.Div(id="api_status_badge",
                         style={"textAlign": "center", "marginTop": "8px"})
            ], className="w-100 text-center")
        ], width=12)
    ], className="py-4 mb-4",
       style={"backgroundColor": "#1A202C",
              "marginRight": "-12px", "marginLeft": "-12px"}),

    # ─── BODY ─────────────────────────────────────────────────────────────
    dbc.Row([

        # ─── Panel izquierdo ──────────────────────────────────────────────
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📥 Ingreso de datos",
                            className="card-title font-weight-bold mb-3",
                            style={"color": "#2C3E50"}),

                    # Departamento
                    html.Label("Departamento",
                               className="small font-weight-bold text-muted"),
                    dcc.Dropdown(
                        id="departamento",
                        options=[{"label": x, "value": x} for x in DEPARTAMENTOS],
                        value="ANTIOQUIA" if DEPARTAMENTOS else None,
                        clearable=False,
                        className="mb-3"
                    ),

                    # Municipio (cascada)
                    html.Label("Municipio",
                               className="small font-weight-bold text-muted"),
                    dcc.Dropdown(id="municipio", clearable=False,
                                 className="mb-3"),

                    # Año y semana
                    dbc.Row([
                        dbc.Col([
                            html.Label("Año",
                                       className="small font-weight-bold text-muted"),
                            dcc.Dropdown(
                                id="anio",
                                options=[{"label": str(y), "value": y}
                                         for y in [2022, 2023, 2024, 2025]],
                                value=DEFAULT_ANIO,
                                clearable=False,
                            ),
                        ], width=5),
                        dbc.Col([
                            html.Label("Semana",
                                       className="small font-weight-bold text-muted"),
                            dcc.Dropdown(
                                id="semana",
                                options=[{"label": f"Sem {i}", "value": i}
                                         for i in range(1, 53)],
                                value=21,
                                clearable=False,
                            ),
                        ], width=7),
                    ], className="mb-3"),

                    # ── Casos ─────────────────────────────────────────────
                    html.Hr(className="my-2"),
                    html.Label("📊 Casos reportados",
                               className="small font-weight-bold text-muted mb-2"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Actual",
                                       className="small text-muted"),
                            dbc.Input(id="casos_actual", type="number",
                                      value=2, min=0),
                        ], width=6),
                        dbc.Col([
                            html.Label("Hace 1 sem",
                                       className="small text-muted"),
                            dbc.Input(id="casos_1", type="number",
                                      value=3, min=0),
                        ], width=6),
                    ], className="mb-2"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Hace 2 sem",
                                       className="small text-muted"),
                            dbc.Input(id="casos_2", type="number",
                                      value=1, min=0),
                        ], width=6),
                        dbc.Col([
                            html.Label("Hace 3 sem",
                                       className="small text-muted"),
                            dbc.Input(id="casos_3", type="number",
                                      value=2, min=0),
                        ], width=6),
                    ], className="mb-3"),

                    # ── Clima ─────────────────────────────────────────────
                    html.Hr(className="my-2"),
                    html.Label("🌡️ Variables climáticas (auto-pobladas)",
                               className="small font-weight-bold text-muted mb-2"),

                    html.Label("Temperatura (°C)",
                               className="small text-muted"),
                    dbc.Input(id="temp_media_c", type="number",
                              step=0.1, className="mb-2"),

                    html.Label("Humedad (%)",
                               className="small text-muted"),
                    dbc.Input(id="humedad_pct", type="number",
                              step=0.1, className="mb-2"),

                    html.Label("Precipitación (mm)",
                               className="small text-muted"),
                    dbc.Input(id="precip_mm", type="number",
                              step=0.1, className="mb-3"),

                    # Botón
                    dbc.Button("🚀 PREDECIR ALERTA",
                               id="btn_predecir",
                               color="danger", size="lg",
                               className="w-100 font-weight-bold")
                ])
            ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                      "borderRadius": "12px"}, className="mb-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H6("ℹ️ DOBLE CLASIFICACIÓN",
                            className="font-weight-bold text-primary mb-2",
                            style={"fontSize": "12px"}),
                    html.P([
                        html.Strong("Zona INS: "),
                        "convención epidemiológica oficial (canal endémico). ",
                        html.Br(),
                        html.Strong("Nivel de Urgencia: "),
                        "escala operacional para tomadores de decisión.",
                        html.Br(),
                        html.Br(),
                        html.Small([
                            "Ambas derivan de la misma probabilidad pero ",
                            "atienden audiencias distintas."
                        ], className="text-muted")
                    ], className="small text-muted m-0",
                       style={"fontSize": "11.5px", "textAlign": "justify"})
                ])
            ], style={"backgroundColor": "#EBF0FF", "border": "none",
                      "borderRadius": "10px"}, className="mt-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H6("⚙️ ARQUITECTURA",
                            className="font-weight-bold mb-2",
                            style={"fontSize": "12px", "color": "#2B6CB0"}),
                    html.P([
                        "Dashboard → API REST → XGBoost. Variables ",
                        "climáticas pre-pobladas desde catálogo histórico, ",
                        "editables por el usuario.",
                        html.Br(),
                        html.Br(),
                        html.Strong("API: "),
                        html.Code(API_URL, style={"fontSize": "10px"})
                    ], className="small text-muted m-0",
                       style={"fontSize": "11.5px"})
                ])
            ], style={"backgroundColor": "#F0FFF4", "border": "none",
                      "borderRadius": "10px"}, className="mt-3"),

        ], width=3),

        # ─── Panel derecho ────────────────────────────────────────────────
        dbc.Col([

            # KPIs - 4 tarjetas
            dbc.Row([
                dbc.Col([crear_tarjeta_kpi("Probabilidad", "—",
                                           "info", "kpi_probabilidad")], width=3),
                dbc.Col([crear_tarjeta_kpi("Nivel Urgencia", "PRESIONA PREDECIR",
                                           "danger", "kpi_urgencia")], width=3),
                dbc.Col([crear_tarjeta_kpi("Zona INS", "—",
                                           "warning", "kpi_zona")], width=3),
                dbc.Col([crear_tarjeta_kpi("Predicción Modelo", "—",
                                           "success", "kpi_prediccion")], width=3),
            ], className="g-3 mb-4"),

            # Error box
            html.Div(id="error_box"),

            # Gráficos
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="grafica_prob", style={"height": "320px"})
                        ])
                    ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                              "borderRadius": "10px"})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="mapa", style={"height": "320px"})
                        ])
                    ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                              "borderRadius": "10px"})
                ], width=6),
            ], className="mb-4"),

            # Recomendaciones
            dbc.Card([
                dbc.CardBody([
                    html.H6("📋 RECOMENDACIONES OPERACIONALES",
                            className="font-weight-bold text-muted mb-3"),
                    html.Div(id="seccion_recomendaciones",
                             className="small text-muted")
                ])
            ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                      "borderRadius": "10px", "minHeight": "180px"},
               className="mb-3"),

            # Disclaimer
            dbc.Card([
                dbc.CardBody([
                    html.H6("ℹ️ INTERPRETACIÓN",
                            className="font-weight-bold mb-2",
                            style={"fontSize": "12px", "color": "#2B6CB0"}),
                    html.P(
                        "El nivel de urgencia combina los casos recientes con "
                        "el riesgo estructural del municipio (clima, altitud, "
                        "densidad poblacional, histórico epidémico). Esta "
                        "herramienta es un apoyo para la toma de decisiones; "
                        "los resultados deben ser interpretados por profesionales "
                        "de salud pública.",
                        className="small text-muted m-0",
                        style={"fontSize": "11.5px", "textAlign": "justify"}
                    )
                ])
            ], style={"backgroundColor": "#FFECEB", "border": "none",
                      "borderRadius": "10px"})

        ], width=9)
    ])
], fluid=True, style={"backgroundColor": "#F4F6F9", "minHeight": "100vh"})


# ──────────────────────────────────────────────────────────────────────────
# CALLBACK 1: cascada departamento → municipio
# ──────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("municipio", "options"),
     Output("municipio", "value")],
    Input("departamento", "value"),
)
def actualizar_municipios(depto):
    if not depto or df_divipola.empty:
        return [], None

    mask = df_divipola["nom_departamento"] == depto
    municipios_depto = df_divipola[mask].sort_values("nom_municipio")
    options = [
        {"label": row["nom_municipio"], "value": row["cod_municipio"]}
        for _, row in municipios_depto.iterrows()
    ]
    default_value = options[0]["value"] if options else None
    return options, default_value


# ──────────────────────────────────────────────────────────────────────────
# CALLBACK 2: VLOOKUP automático de clima al cambiar municipio o semana
# ──────────────────────────────────────────────────────────────────────────

@app.callback(
    [Output("temp_media_c", "value"),
     Output("humedad_pct", "value"),
     Output("precip_mm", "value")],
    [Input("departamento", "value"),
     Input("semana", "value")],
)
def autollenar_clima(depto, semana):
    """Busca el clima del departamento en la semana solicitada."""
    if df_clima.empty or not depto or not semana:
        return 24.0, 78.0, 45.0  # defaults

    depto_norm = normalize_depto(depto)
    match = df_clima[
        (df_clima["depto_norm"] == depto_norm) &
        (df_clima["semana_epi"] == int(semana))
    ]

    if match.empty:
        return 24.0, 78.0, 45.0

    row = match.iloc[0]
    return (
        round(float(row["temp_media_c"]), 1),
        round(float(row["humedad_pct"]), 1),
        round(float(row["precip_mm"]), 1),
    )


# ──────────────────────────────────────────────────────────────────────────
# CALLBACK 3: badge de estado API
# ──────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("api_status_badge", "children"),
    Input("departamento", "value"),
)
def mostrar_estado_api(_):
    health = check_api_health()
    if "error" in health:
        return dbc.Badge(f"⚠️ API offline: {health['error']}",
                         color="danger", className="me-1")
    return dbc.Badge(
        f"✓ API online · Modelo v{health.get('model_version', '?')}",
        color="success", className="me-1"
    )


# ──────────────────────────────────────────────────────────────────────────
# CALLBACK 4: predicción
# ──────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("kpi_probabilidad", "children"),
        Output("kpi_urgencia", "children"),
        Output("kpi_zona", "children"),
        Output("kpi_prediccion", "children"),
        Output("grafica_prob", "figure"),
        Output("mapa", "figure"),
        Output("seccion_recomendaciones", "children"),
        Output("error_box", "children"),
    ],
    Input("btn_predecir", "n_clicks"),
    [
        State("departamento", "value"),
        State("municipio", "value"),
        State("anio", "value"),
        State("semana", "value"),
        State("casos_actual", "value"),
        State("casos_1", "value"),
        State("casos_2", "value"),
        State("casos_3", "value"),
        State("temp_media_c", "value"),
        State("humedad_pct", "value"),
        State("precip_mm", "value"),
    ],
    prevent_initial_call=False,
)
def predecir(n, depto, cod_mun, anio, semana, c0, c1, c2, c3,
             temp, humedad, precip):

    # Estado inicial
    if not n:
        empty_fig = px.scatter(title="Presiona 'PREDECIR ALERTA' para empezar")
        empty_fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        return ("—", "PRESIONA PREDECIR", "—", "—",
                empty_fig, empty_fig,
                html.P("Configura los datos y presiona el botón.",
                       className="text-muted"),
                None)

    # Validar inputs
    if not cod_mun:
        return ("—", "ERROR", "—", "—",
                px.scatter(), px.scatter(),
                html.P("Selecciona un municipio.", className="text-danger"),
                dbc.Alert("Falta el municipio.", color="warning"))

    # Obtener metadatos del municipio
    mun_row = df_divipola[df_divipola["cod_municipio"] == cod_mun]
    if mun_row.empty:
        return ("—", "ERROR", "—", "—",
                px.scatter(), px.scatter(),
                html.P("Municipio no encontrado.", className="text-danger"),
                dbc.Alert(f"Municipio {cod_mun} no en DIVIPOLA.",
                          color="danger"))

    nom_mun = mun_row.iloc[0]["nom_municipio"]
    latitud = float(mun_row.iloc[0]["latitud"])
    longitud = float(mun_row.iloc[0]["longitud"])

    # Construir payload
    payload = {
        "cod_municipio": cod_mun,
        "anio": int(anio),
        "semana_epi": int(semana),
        "casos_actual": float(c0 or 0),
        "casos_lag1": float(c1 or 0),
        "casos_lag2": float(c2 or 0),
        "casos_lag3": float(c3 or 0),
        "temp_media_c": float(temp or 24.0),
        "humedad_pct": float(humedad or 78.0),
        "precip_mm": float(precip or 45.0),
    }

    # Llamar API
    result = predict_via_api(payload)

    if "error" in result:
        return ("—", "ERROR", "—", "—",
                px.scatter(title="Error"), px.scatter(title="Error"),
                html.P("No se pudo predecir.", className="text-danger"),
                dbc.Alert([
                    html.H6("❌ Error en la API", className="alert-heading"),
                    html.P(result["error"]),
                ], color="danger", dismissable=True))

    # Procesar respuesta
    prob = float(result.get("probability", 0))
    pred = int(result.get("prediction", 0))
    label_modelo = result.get("label", "DESCONOCIDO")

    # Clasificar en las dos escalas
    urgencia = clasificar_nivel_urgencia(prob)
    zona = clasificar_zona_ins(prob, c0 or 0)

    # KPIs
    kpi_prob = f"{prob:.1%}"
    kpi_urgencia = f"{urgencia['emoji']} {urgencia['nivel']}"
    kpi_zona = f"{zona['emoji']} {zona['zona']}"
    kpi_pred = f"⚠️ {label_modelo}" if pred == 1 else f"✅ {label_modelo}"

    # ── Gráfica de probabilidad con todos los umbrales ────────────────────
    fig_prob = px.bar(
        x=[prob], y=[f"{nom_mun}<br>Sem {semana}/{anio}"],
        orientation="h", range_x=[0, 1],
        labels={"x": "Probabilidad de alerta", "y": ""},
        title=f"Probabilidad — {nom_mun}",
    )
    bar_color = "#E53E3E" if prob >= 0.75 else \
                "#ED8936" if prob >= 0.50 else \
                "#ECC94B" if prob >= 0.30 else "#48BB78"
    fig_prob.update_traces(
        marker_color=bar_color,
        text=[f"{prob:.1%}"], textposition="outside",
    )
    fig_prob.add_vline(x=0.30, line_dash="dash", line_color="#ECC94B",
                       annotation_text="Precaución",
                       annotation_position="top")
    fig_prob.add_vline(x=0.50, line_dash="dash", line_color="#ED8936",
                       annotation_text="Media",
                       annotation_position="top")
    fig_prob.add_vline(x=0.75, line_dash="dash", line_color="#E53E3E",
                       annotation_text="Alta",
                       annotation_position="top")
    fig_prob.update_layout(showlegend=False,
                           margin=dict(t=60, b=40, l=20, r=20))

    # ── Mapa ──────────────────────────────────────────────────────────────
    mapa_df = pd.DataFrame({
        "Municipio": [nom_mun],
        "lat": [latitud], "lon": [longitud],
        "Riesgo": [prob],
        "Urgencia": [urgencia["nivel"]],
        "Zona INS": [zona["zona"]],
    })
    fig_mapa = px.scatter_mapbox(
        mapa_df, lat="lat", lon="lon",
        size=[30], color="Riesgo", color_continuous_scale="Reds",
        range_color=[0, 1], zoom=5.2, mapbox_style="carto-positron",
        hover_name="Municipio",
        hover_data={"Riesgo": ":.2%", "Urgencia": True, "Zona INS": True,
                    "lat": False, "lon": False},
        title=f"Ubicación — {nom_mun}, {depto}"
    )
    fig_mapa.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    # ── Recomendaciones diferenciadas ─────────────────────────────────────
    recomendaciones = [
        html.Div([
            html.Strong(f"{urgencia['emoji']} Nivel de Urgencia: ",
                        className="text-dark"),
            html.Span(f"{urgencia['nivel']} — {urgencia['desc']}")
        ], className="mb-2"),
        html.Div([
            html.Strong(f"{zona['emoji']} Zona INS: ",
                        className="text-dark"),
            html.Span(f"{zona['zona']} (clasificación epidemiológica oficial)")
        ], className="mb-2"),
        html.Hr(className="my-2"),
        html.Div([
            html.Strong("🔹 Condiciones consultadas: ",
                        className="text-dark"),
            html.Span(f"{temp}°C, {humedad}% humedad, {precip}mm precipitación.")
        ], className="mb-2"),
        html.Div([
            html.Strong("🔹 Casos reportados: ",
                        className="text-dark"),
            html.Span(f"actual={c0}, hace 1 sem={c1}, hace 2 sem={c2}, hace 3 sem={c3}.")
        ], className="mb-2"),
    ]

    # Acciones según nivel de urgencia
    if urgencia["nivel"] == "ALTA":
        recomendaciones.append(html.Div([
            html.Strong("🚨 ACCIÓN INMEDIATA: ",
                        className="text-danger fw-bold"),
            html.Span("Activar protocolos de fumigación espacial, cercos "
                      "epidemiológicos y comunicación de riesgo masiva. "
                      "Notificar al INS y a Secretaría Departamental de Salud.")
        ], className="mb-2"))
    elif urgencia["nivel"] == "MEDIA":
        recomendaciones.append(html.Div([
            html.Strong("⚠️ VIGILANCIA ACTIVA: ",
                        className="text-warning fw-bold"),
            html.Span("Intensificar búsqueda activa de casos, eliminar "
                      "criaderos potenciales en peridomicilio, reforzar "
                      "uso de toldillos y repelentes en la población.")
        ], className="mb-2"))
    elif urgencia["nivel"] == "PRECAUCIÓN":
        recomendaciones.append(html.Div([
            html.Strong("🟡 MONITOREO AUMENTADO: ",
                        className="text-info fw-bold"),
            html.Span("Verificar inventario de insumos. Mantener "
                      "comunicación preventiva con la comunidad sobre "
                      "eliminación de depósitos de agua.")
        ], className="mb-2"))
    else:
        recomendaciones.append(html.Div([
            html.Strong("✅ VIGILANCIA RUTINARIA: ",
                        className="text-success fw-bold"),
            html.Span("Continuar actividades habituales de control vectorial "
                      "y educación en salud.")
        ], className="mb-2"))

    return (
        kpi_prob, kpi_urgencia, kpi_zona, kpi_pred,
        fig_prob, fig_mapa, recomendaciones, None,
    )


# ──────────────────────────────────────────────────────────────────────────
# ARRANQUE
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
