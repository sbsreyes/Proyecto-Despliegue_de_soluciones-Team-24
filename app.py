# ==========================================================================
# AI-lerta: sistema de alerta temprana para dengue en municipios de Colombia 
# mediante canal endémico, variables climáticas y aprendizaje automático 
# Dashboard con predicción XGBoost
# Proyecto Maestria IA 
# Universidad de los Andes
# Autores: Juan Sebastián Reyes Acosta, Julián Felipe Moncada Castro, 
# Irlanda Katiuzhca Robles Orellana, Amalia Catalina Lehmann Oliveros 
# ==========================================================================

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import joblib


# 1. CARGA DE MODELO Y BASE DE DATOS CLIMÁTICA HISTÓRICA
modelo = joblib.load("models/modelo_xgboost.pkl")

# Intentar cargar el CSV desde assets/
try:
    df_clima_historico = pd.concat([
    pd.read_csv("assets/clima_colombia_2025.csv")
    ], ignore_index=True)
except FileNotFoundError:

    semanas_dummy = []
    deptos_base = {
        "Antioquia": {"altitud": 1500, "poblacion": 7000000, "lat": 6.2442, "lon": -75.5812, "temp_base": 22, "hum_base": 75, "precip_base": 90},
        "Cundinamarca": {"altitud": 2600, "poblacion": 3200000, "lat": 4.7110, "lon": -74.0721, "temp_base": 14, "hum_base": 70, "precip_base": 40},
        "Valle del Cauca": {"altitud": 1000, "poblacion": 4600000, "lat": 3.4516, "lon": -76.5320, "temp_base": 24, "hum_base": 76, "precip_base": 100},
        "Meta": {"altitud": 467, "poblacion": 1142948, "lat": 4.1420, "lon": -73.6266, "temp_base": 26, "hum_base": 78, "precip_base": 110},
        "Arauca": {"altitud": 125, "poblacion": 300000, "lat": 7.0847, "lon": -70.7591, "temp_base": 27, "hum_base": 80, "precip_base": 120},
        "Casanare": {"altitud": 350, "poblacion": 450000, "lat": 5.3378, "lon": -72.3959, "temp_base": 26, "hum_base": 77, "precip_base": 105}
    }
    for d, info in deptos_base.items():
        for sem in range(1, 53):
            semanas_dummy.append({
                "departamento": d,
                "semana_epi": sem,
                "temp_media_c": info["temp_base"],
                "humedad_pct": info["hum_base"],
                "precip_mm": info["precip_base"],
                "altitud_msnm": info["altitud"],
                "poblacion": info["poblacion"],
                "lat": info["lat"],
                "lon": info["lon"]
            })
    df_clima_historico = pd.DataFrame(semanas_dummy)

LISTA_DEPARTAMENTOS = sorted(df_clima_historico["departamento"].unique())

FEATURES = [
    'altitud_msnm', 'cat_altitud_enc', 'anio', 'semana_epi', 'semana_sin', 'semana_cos',
    'temporada_lluvias', 'temp_media_c', 'humedad_pct', 'precip_mm', 'grados_dia',
    'temp_optima', 'temp_letal', 'temp_inhibicion', 'indice_idoneidad', 'precip_acum4',
    'precip_acum8', 'temp_lag2', 'temp_lag3', 'temp_lag4', 'precip_lag2', 'precip_lag3',
    'precip_lag4', 'humedad_lag1', 'humedad_lag2', 'humedad_lag3', 'casos_lag1',
    'casos_lag2', 'casos_lag3', 'casos_lag4', 'casos_ma4', 'casos_tendencia', 'poblacion',
    'año_epidemico', 'post_epidemia',
]


# CONFIGURACIÓN DE LA APP Y COMPONENTES
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

def crear_tarjeta_kpi(titulo, valor, color_borde="primary", id_dinamico=None):
    return dbc.Card([
        dbc.CardBody([
            html.H6(titulo, className="text-muted text-uppercase font-weight-bold mb-1", style={"fontSize": "11px"}),
            html.H3(valor, id=id_dinamico, className="font-weight-bold m-0", style={"color": "#2C3E50", "fontSize": "24px"})
        ], className="p-3")
    ], style={
        "borderTop": f"4px solid var(--bs-{color_borde})", 
        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
        "height": "100%",
        "borderRadius": "8px"
    })

# PRINCIPAL
app.layout = dbc.Container([
    
    # 1. ENCABEZADO SUPERIOR
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🦟 AI-LERTA DENGUE COLOMBIA", style={"textAlign": "center", "fontWeight": "bold", "color": "#00E5FF", "margin": "0"}),
                html.H5("Sistema Inteligente de Alertas Tempranas", style={"textAlign": "center", "color": "lightgray", "margin": "5px 0 0 0", "fontSize": "16px"})
            ], className="w-100 text-center")
        ], width=12)
    ], className="py-4 mb-4", style={"backgroundColor": "#1A202C", "marginRight": "-12px", "marginLeft": "-12px"}),

    # 2. CUERPO PRINCIPAL DEL DASHBOARD
    dbc.Row([
        
        # PANEL IZQUIERDO
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📥 Ingreso de datos", className="card-title font-weight-bold mb-3", style={"color": "#2C3E50"}),
                    
                    html.Label("Departamento", className="small font-weight-bold text-muted"),
                    dcc.Dropdown(
                        id="departamento",
                        options=[{"label": x, "value": x} for x in LISTA_DEPARTAMENTOS],
                        value="Antioquia",
                        clearable=False,
                        className="mb-3"
                    ),

                    html.Label("Semana epidemiológica", className="small font-weight-bold text-muted"),
                    dcc.Dropdown(
                        id="semana",
                        options=[{"label": f"Semana {i}", "value": i} for i in range(1, 53)],
                        value=21,
                        clearable=False,
                        className="mb-3"
                    ),

                    # VALORES
                    html.Label("Casos semana actual", className="small font-weight-bold text-muted"),
                    dbc.Input(id="casos_actual", type="number", value=2, className="mb-3"),

                    html.Label("Casos hace 1 semana", className="small font-weight-bold text-muted"),
                    dbc.Input(id="casos_1", type="number", value=3, className="mb-3"),

                    html.Label("Casos hace 2 semanas", className="small font-weight-bold text-muted"),
                    dbc.Input(id="casos_2", type="number", value=1, className="mb-3"),

                    html.Label("Casos hace 3 semanas", className="small font-weight-bold text-muted"),
                    dbc.Input(id="casos_3", type="number", value=2, className="mb-4"),

                    dbc.Button("🚀 PREDECIR ALERTA", id="btn_predecir", color="danger", size="lg", className="w-100 font-weight-bold mb-3")
                ])
            ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)", "borderRadius": "12px"}, className="mb-3"),
            
            # EXPLICACIÓN
            dbc.Card([
                dbc.CardBody([
                    html.H6("⚙️ ¿CÓMO FUNCIONA?", className="font-weight-bold text-success mb-2", style={"fontSize": "12px", "color": "#2B6CB0"}),
                    html.P(
                        "El sistema utiliza datos históricos de casos de dengue, clima y población "
                        "para predecir la probabilidad de incrementos significativos en los casos "
                        "en las próximas semanas.", 
                        className="small text-muted m-0", 
                        style={"fontSize": "11.5px", "textAlign": "justify"}
                    )
                ])
            ], style={"backgroundColor": "#F0FFF4", "border": "none", "borderRadius": "10px"}, className="mt-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H6("ℹ️ INFRAESTRUCTURA CLIMÁTICA", className="font-weight-bold text-primary mb-2", style={"fontSize": "12px"}),
                    html.P("El sistema extrae dinámicamente las condiciones meteorológicas del CSV según la semana elegida, ajustando las variables climáticas de forma transparente antes de la predicción.", className="small text-muted m-0", style={"fontSize": "11.5px", "textAlign": "justify"})
                ])
            ], style={"backgroundColor": "#EBF0FF", "border": "none", "borderRadius": "10px"}, className="mt-3")

        ], width=3),

        # PANEL DERECHO
        dbc.Col([
            
            # FILA DE TARJETAS KPI
            dbc.Row([
                dbc.Col([crear_tarjeta_kpi("Probabilidad", "0.0%", "info", "kpi_probabilidad")], width=3),
                dbc.Col([crear_tarjeta_kpi("Nivel", "CALCULANDO...", "danger", "kpi_riesgo")], width=3),
                dbc.Col([crear_tarjeta_kpi("Casos Actuales", "0", "warning", "kpi_casos")], width=3),
                dbc.Col([crear_tarjeta_kpi("Tendencia", "ESTABLE", "success", "kpi_tendencia")], width=3),
            ], className="g-3 mb-4"),
            
            # FILA DE GRÁFICOS Y MAPAS
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="grafica", style={"height": "340px"})
                        ])
                    ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)", "borderRadius": "10px"})
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="mapa", style={"height": "340px"})
                        ])
                    ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)", "borderRadius": "10px"})
                ], width=6),
            ], className="mb-4"),

            # RECOMENDACIONES
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📋 RECOMENDACIONES ESTRATÉGICAS DE SALUD", className="font-weight-bold text-muted mb-3"),
                            html.Div(id="seccion_recomendaciones", className="small text-muted")
                        ])
                    ], style={"boxShadow": "0 4px 6px rgba(0,0,0,0.05)", "borderRadius": "10px","height": "280px"}, className="mt-3")
                ], width=12)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("ℹ️ ¡TENER EN CUENTA!", className="font-weight-bold text-success mb-2", style={"fontSize": "12px", "color": "#2B6CB0"}),
                            html.P(
                                "Este sistema es una herramienta de apoyo para la toma de decisiones en salud pública. "
                                "Los resultados deben ser interpretados por profesionales de la salud. ",
                                className="small text-muted m-0", 
                                style={"fontSize": "11.5px", "textAlign": "justify"}
                            )
                        ])
                    ], style={"backgroundColor": "#FFECEB", "border": "none", "borderRadius": "10px"}, className="mt-3")
                ], width=12)
            ])
            
        ], width=9)
    ])
], fluid=True, style={"backgroundColor": "#F4F6F9", "minHeight": "100vh"})


# CALLBACK - FILTRADO CLIMÁTICO
@app.callback(
    [
        Output("kpi_probabilidad", "children"),
        Output("kpi_riesgo", "children"),
        Output("kpi_casos", "children"),
        Output("kpi_tendencia", "children"),
        Output("grafica", "figure"),
        Output("mapa", "figure"),
        Output("seccion_recomendaciones", "children")
    ],
    Input("btn_predecir", "n_clicks"),
    [
        State("departamento", "value"),
        State("semana", "value"),
        State("casos_actual", "value"),
        State("casos_1", "value"),
        State("casos_2", "value"),
        State("casos_3", "value"),
    ]
)
def evaluar_modelo_dinamico(n, depto, semana, c0, c1, c2, c3):
    if c0 is None or c1 is None:
        c0, c1, c2, c3 = 2, 3, 1, 2

    # Búsqueda directa en el DataFrame
    registro_clima = df_clima_historico[
        (df_clima_historico["departamento"] == depto) & 
        (df_clima_historico["semana_epi"] == semana)
    ]
    
    if not registro_clima.empty:
        clima = registro_clima.iloc[0]
        temp_media = clima["temp_media_c"]
        humedad = clima["humedad_pct"]
        precip = clima["precip_mm"]
        altitud = clima["altitud_msnm"]
        poblacion = clima["poblacion"]
        latitud = clima["lat"]
        longitud = clima["lon"]
    else:
        temp_media, humedad, precip, altitud, poblacion, latitud, longitud = 22, 75, 90, 1500, 7000000, 6.2442, -75.5812

    anio = 2024

    # Ingeniería de Variables
    semana_sin = np.sin(2 * np.pi * semana / 52)
    semana_cos = np.cos(2 * np.pi * semana / 52)
    temporada_lluvias = 1 if ((13 <= semana <= 22) or (35 <= semana <= 44)) else 0
    grados_dia = max(temp_media - 18, 0)
    temp_optima = int(26 <= temp_media <= 29)
    temp_letal = int(temp_media > 35)
    temp_inhibicion = int(temp_media < 16)
    indice_idoneidad = temp_media * (1 / (1 + np.exp(0.003 * (altitud - 2200))))
    casos_ma4 = np.mean([c0, c1, c2, c3])
    casos_tendencia = c0 - c1

    fila = pd.DataFrame([{
        'altitud_msnm': altitud, 'cat_altitud_enc': 1, 'anio': anio, 'semana_epi': semana,
        'semana_sin': semana_sin, 'semana_cos': semana_cos, 'temporada_lluvias': temporada_lluvias,
        'temp_media_c': temp_media, 'humedad_pct': humedad, 'precip_mm': precip,
        'grados_dia': grados_dia, 'temp_optima': temp_optima, 'temp_letal': temp_letal, 'temp_inhibicion': temp_inhibicion,
        'indice_idoneidad': indice_idoneidad, 'precip_acum4': precip * 4, 'precip_acum8': precip * 8,
        'temp_lag2': temp_media, 'temp_lag3': temp_media, 'temp_lag4': temp_media,
        'precip_lag2': precip, 'precip_lag3': precip, 'precip_lag4': precip,
        'humedad_lag1': humedad, 'humedad_lag2': humedad, 'humedad_lag3': humedad,
        'casos_lag1': c1, 'casos_lag2': c2, 'casos_lag3': c3, 'casos_lag4': c3, 'casos_ma4': casos_ma4,
        'casos_tendencia': casos_tendencia, 'poblacion': poblacion, 'año_epidemico': 0, 'post_epidemia': 0,
    }])
    fila = fila[FEATURES]

    # Modelo XGBoost
    prob = modelo.predict_proba(fila)[0][1]
    
    # REGLA DE NEGOCIO: Evita falsas alertas si el volumen de casos históricos de las 4 semanas es insignificante
    if casos_ma4 < 10:
        prob = max(0.04, prob * (casos_ma4 / 15))

    # Definición de umbrales
    if prob >= 0.75:
        nivel = "🔴 ALTO"
    elif prob >= 0.35:
        nivel = "🟡 MEDIO"
    else:
        nivel = "🟢 BAJO"

    txt_tendencia = "EN AUMENTO 📈" if casos_tendencia > 0 else "EN DESCENSO 📉" if casos_tendencia < 0 else "ESTABLE ➡️"

    # Gráfica de Tendencia
    serie = pd.DataFrame({
        "Semana": ["Hace 3", "Hace 2", "Hace 1", "Actual"],
        "Casos": [c3, c2, c1, c0]
    })
    fig = px.line(serie, x="Semana", y="Casos", markers=True, title=f"Tendencia de Casos - {depto}")

    # MAPA GEOGRAFICO
    mapa_df = pd.DataFrame({
        "Departamento": [depto],
        "lat": [latitud],
        "lon": [longitud],
        "Riesgo Predicho": [prob],
        "Temperatura": [f"{temp_media:.1f}°C"],
        "Lluvias": [f"{precip:.1f} mm"]
    })

    fig_mapa = px.scatter_mapbox(
        mapa_df,
        lat="lat",
        lon="lon",
        size=[25],
        color="Riesgo Predicho",
        color_continuous_scale="Reds",
        range_color=[0, 1],
        zoom=5.2,
        mapbox_style="carto-positron",
        hover_name="Departamento",
        hover_data={
            "Riesgo Predicho": ":.2%", 
            "Temperatura": True,
            "Lluvias": True,
            "lat": False, 
            "lon": False
        },
        title=f"Ubicación del Riesgo - {depto}"
    )
    fig_mapa.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    # Recomendaciones
    recomendaciones = [
        html.Div([
            html.Strong("🔹 Monitoreo del Clima Local: ", className="text-dark"),
            html.Span(f"Condiciones actuales: {temp_media:.1f}°C, Humedad: {humedad:.1f}%, Precipitación: {precip:.1f}mm.")
        ], className="mb-2"),
        html.Div([
            html.Strong("🔹 Medidas Preventivas: ", className="text-dark"),
            html.Span("Eliminación de criaderos potenciales y depósitos descubiertos en el peridomicilio.")
        ], className="mb-2")
    ]
    
    if prob >= 0.35:
        recomendaciones.append(html.Div([
            html.Strong("⚠️ Acciones Sanitarias: ", className="text-warning"),
            html.Span("Presencia estacional del vector por idoneidad térmica. Reforzar uso de toldillos en la población.")
        ], className="mb-2"))
    if prob >= 0.75:
        recomendaciones.append(html.Div([
            html.Strong("🚨 Alerta Sanitaria Temprana: ", className="text-danger"),
            html.Span("Riesgo crítico calculado. Activación de protocolos de fumigación espacial y cercos epidemiológicos.")
        ]))

    return (
        f"{prob:.1%}", 
        nivel, 
        f"{c0}", 
        txt_tendencia, 
        fig, 
        fig_mapa,
        recomendaciones
    )

if __name__ == "__main__":
    # Ejecución Local
    #app.run(debug=True)

    # Ejecución AWS
    app.run_server(host="0.0.0.0", port=8050, debug=False)