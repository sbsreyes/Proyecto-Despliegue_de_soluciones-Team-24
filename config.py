# ── Configuración global del proyecto ─────────────────────────────────────────
# Equipo 24 — Predicción de Dengue en Colombia

# ── AWS S3 ────────────────────────────────────────────────────────────────────
BUCKET          = 'dengue-colombia-team24'
RUTA_SIVIGILA   = f's3://{BUCKET}/datos/sivigila/'
RUTA_ALTITUD    = f's3://{BUCKET}/datos/altitud/municipios_colombia_altitud.csv'
RUTA_DANE       = f's3://{BUCKET}/datos/dane/'
RUTA_GEE        = f's3://{BUCKET}/datos/gee/'          # pendiente
RUTA_OUTPUTS    = f's3://{BUCKET}/outputs/'

# ── Períodos de análisis ──────────────────────────────────────────────────────
AÑOS_TRAIN      = list(range(2009, 2020))   # 2009–2019
AÑOS_SKIP       = [2020, 2021]              # pandemia COVID-19
AÑOS_TEST       = [2022, 2023, 2024]        # test

# ── Códigos SIVIGILA dengue ───────────────────────────────────────────────────
CODIGOS_DENGUE  = [210, 220, 580]
# 210 = Dengue (sin signos / con signos de alarma)
# 220 = Dengue grave
# 580 = Mortalidad por dengue

# ── Umbrales altitud Aedes aegypti (literatura colombiana) ────────────────────
# Fuente: Suárez & Nelson (1981), INS Colombia
ALTITUD_BAJO    = 1000    # < 1.000 msnm    — riesgo alto
ALTITUD_MEDIO   = 1800    # 1.000–1.800 msnm — riesgo alto
ALTITUD_ALTO    = 2200    # 1.800–2.200 msnm — riesgo moderado
                          # > 2.200 msnm    — riesgo bajo

# ── URLs fuentes de datos ─────────────────────────────────────────────────────
URL_DIVIPOLA    = 'https://www.datos.gov.co/api/views/vafm-j2df/rows.csv?accessType=DOWNLOAD'
URL_DANE_05_17  = 'https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx'
URL_DANE_18_42  = 'https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/PPED-AreaMun-2018-2042_VP.xlsx'
FILE_I