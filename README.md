# Predicción de Dengue en Colombia 🦟

## Descripción
Modelo predictivo de casos de dengue a nivel municipal en Colombia, 
integrando datos de SIVIGILA, altitud municipal (SRTM), población (DANE) 
y variables climáticas (GEE).

## Equipo — Team 24
| Integrante | Modelo | Rama |
|---|---|---|
| Cata Lehmann | Binomial Negativa + XGBoost + BiLSTM | feature/modelo-dengue-lehmann |
| Compañero 2 | Por definir | Por definir |
| Compañero 3 | Por definir | Por definir |
| Compañero 4 | Por definir | Por definir |

## Datos
| Fuente | Variable | Período |
|---|---|---|
| SIVIGILA (INS) | Casos de dengue por municipio | 2009–2024 |
| DANE | Proyecciones de población municipal | 2009–2024 |
| SRTM/Open-Elevation | Altitud municipal (msnm) | Estático |
| Google Earth Engine | Temperatura, precipitación, humedad | Pendiente |

## Estructura del repositorio
lehmann/               → Modelos de Ana Lehmann
shared/                → Funciones compartidas
data/                  → Datos (no versionados en Git)
outputs/               → Resultados y gráficas
config.py              → Configuración global
requirements.txt       → Dependencias Python

## División train/test
- Train: 2009–2019 (11 años, pre-pandemia)
- Skip:  2020–2021 (pandemia COVID-19)
- Test:  2022–2024

## Cómo ejecutar
1. Clonar el repositorio
   git clone https://github.com/sbsreyes/Proyecto-Despliegue_de_soluciones-Team-24.git

2. Instalar dependencias
   pip install -r requirements.txt

3. Ejecutar notebooks en orden
   lehmann/01_altitud_municipios.ipynb
   lehmann/02_sivigila_pipeline.ipynb
   lehmann/03_pipeline_modelo_final.ipynb
   lehmann/04_modelo_binomial_negativa.ipynb
   lehmann/05_modelo_xgboost.ipynb
   lehmann/06_modelo_bilstm.ipynb

## Referencias
- Cheng et al. (2026). Dengue fever prediction based on meteorological 
  features and deep learning models. Infectious Disease Modelling, 11, 683-700.
- INS Colombia. SIVIGILA - Sistema Nacional de Vigilancia en Salud Publica.
  https://www.ins.gov.co/Direcciones/Vigilancia/Paginas/SIVIGILA.aspx
- DANE. Proyecciones de poblacion municipal 2005-2042.
  https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion
- IGAC / NASA SRTM. Altitud municipal Colombia (30m resolucion).
  Procesado via Open-Elevation API (https://api.open-elevation.com)
  y DIVIPOLA (https://www.datos.gov.co/resource/vafm-j2df)
- Google Earth Engine (GEE). Variables climaticas: temperatura, 
  precipitacion y humedad relativa. (pendiente de integracion)
  https://earthengine.google.com