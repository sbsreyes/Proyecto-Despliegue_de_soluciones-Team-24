# 🦟 AI-lerta Dengue Colombia — Despliegue

API REST para predicción de alertas de dengue por municipio y semana epidemiológica.
Sistema basado en canal endémico municipal + variables climáticas + biología del vector *Aedes aegypti*.

---

## 📁 Estructura

```
dengue-deployment/
│
├── model-package/         ← PASO 1: empaquetar el modelo
│   ├── tox.ini
│   ├── setup.py
│   ├── model/
│   │   ├── config.yml
│   │   ├── pipeline.py
│   │   ├── train_pipeline.py
│   │   ├── predict.py
│   │   ├── processing/    ← FeatureEngineer + validation + data_manager
│   │   ├── datasets/      ← dataset_dengue_completo.csv
│   │   └── trained/       ← .pkl generado por tox
│   └── tests/
│
└── api-docker/            ← PASO 2: API en Docker
    ├── Dockerfile
    ├── docker-compose.yml
    ├── packages/          ← AQUÍ va el .whl del Paso 1
    └── app/
        ├── main.py
        ├── core/config.py
        ├── schemas/predict.py
        └── api/endpoints/ (health.py + predict.py)
```

---

## 🔧 Prerequisitos

- Python 3.11+
- Docker + docker-compose
- El archivo `dataset_dengue_completo.csv` (descargar de Drive)

---

## 🚀 PASO 1 — Empaquetar el modelo (local)

```bash
cd dengue-deployment/model-package/

# 1.1 Colocar el dataset
# Copia dataset_dengue_completo.csv aquí:
#   model/datasets/dataset_dengue_completo.csv

# 1.2 Ambiente virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install --upgrade pip
pip install tox build

# 1.3 Entrenar
tox run -e train

# 1.4 Tests
tox run -e test_package

# 1.5 Construir el .whl
python3 -m build
# → genera dist/dengue_model-0.0.1-py3-none-any.whl

# 1.6 Copiar el .whl a la API
cp dist/dengue_model-0.0.1-py3-none-any.whl ../api-docker/packages/
```

---

## 🐳 PASO 2 — Levantar la API (local con Docker)

```bash
cd ../api-docker/

# 2.1 Construir y arrancar
docker-compose up --build

# En background:
docker-compose up -d --build

# Ver logs:
docker-compose logs -f
```

**API disponible en:**
- `http://localhost:8001/docs` — Swagger UI interactivo
- `http://localhost:8001/api/v1/health` — Health check
- `http://localhost:8001/api/v1/predict` — Predicción batch
- `http://localhost:8001/api/v1/predict/single` — Predicción individual

---

## 🧪 Verificar que funciona

### Health check
```bash
curl http://localhost:8001/api/v1/health
```

Respuesta esperada:
```json
{
  "status": "ok",
  "model_version": "0.0.1",
  "api_version": "1.0.0"
}
```

### Predicción individual
```bash
curl -X POST http://localhost:8001/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "cod_municipio": "05001",
    "anio": 2024,
    "semana_epi": 32,
    "altitud_msnm": 1523.0,
    "cat_altitud": "Medio (1.000-1.800 m)",
    "poblacion": 2351077.0,
    "temp_media_c": 22.5,
    "humedad_pct": 78.0,
    "precip_mm": 45.3
  }'
```

Respuesta:
```json
{
  "index": 0,
  "cod_municipio": "05001",
  "anio": 2024,
  "semana_epi": 32,
  "prediction": 1,
  "probability": 0.8742,
  "label": "ALERTA"
}
```

### Predicción batch
```bash
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"cod_municipio": "05001", "anio": 2024, "semana_epi": 32},
      {"cod_municipio": "76001", "anio": 2024, "semana_epi": 32}
    ]
  }'
```

---

## ☁️ PASO 3 — Despliegue en AWS EC2

### 3.1 Crear EC2

1. AWS Console → EC2 → Launch Instance
2. **Name:** `dengue-api`
3. **AMI:** Ubuntu Server 22.04 LTS (free tier elegible)
4. **Instance type:** `t3.small` (2 vCPU, 2 GB RAM) — necesitas más de t2.micro porque Docker + Python + XGBoost
5. **Key pair:** crear nuevo o usar existente (descarga el .pem)
6. **Network settings:**
   - VPC: default
   - Public IP: enable
   - **Security group: crear nuevo** llamado `dengue-api-sg`:
     - SSH (port 22) — desde Mi IP
     - Custom TCP (port 8001) — desde 0.0.0.0/0 (público para el dashboard)
     - HTTP (port 80) — desde 0.0.0.0/0 (opcional)
7. **Storage:** 16 GB gp3
8. Launch instance

### 3.2 Conectarse y preparar la EC2

```bash
# Desde tu máquina local:
chmod 400 ~/Downloads/tu-keypair.pem
ssh -i ~/Downloads/tu-keypair.pem ubuntu@<EC2-PUBLIC-IP>
```

Una vez dentro:
```bash
# Actualizar
sudo apt update && sudo apt upgrade -y

# Instalar Docker
sudo apt install -y docker.io docker-compose-v2 git
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker

# Verificar
docker --version
docker compose version
```

### 3.3 Subir el código a EC2

**Opción A — Git (recomendado):**
```bash
git clone https://github.com/sbsreyes/Proyecto-Despliegue_de_soluciones-Team-24.git
cd Proyecto-Despliegue_de_soluciones-Team-24/dengue-deployment/api-docker
```

**Opción B — SCP desde tu máquina:**
```bash
# Desde tu máquina local
scp -i ~/Downloads/tu-keypair.pem -r dengue-deployment ubuntu@<EC2-IP>:/home/ubuntu/
```

### 3.4 Subir el .whl a packages/

```bash
# Desde tu máquina local
scp -i ~/Downloads/tu-keypair.pem \
    model-package/dist/dengue_model-0.0.1-py3-none-any.whl \
    ubuntu@<EC2-IP>:/home/ubuntu/dengue-deployment/api-docker/packages/
```

### 3.5 Arrancar la API en EC2

```bash
cd dengue-deployment/api-docker
docker compose up -d --build
docker compose logs -f
```

### 3.6 Probar desde fuera

```bash
# Desde tu máquina local:
curl http://<EC2-PUBLIC-IP>:8001/api/v1/health
```

**URL pública para el equipo del dashboard:**
```
http://<EC2-PUBLIC-IP>:8001/api/v1/predict/single
```

---

## 📞 Para el equipo del Dashboard

### Endpoint a consumir
```
POST http://<EC2-PUBLIC-IP>:8001/api/v1/predict/single
Content-Type: application/json
```

### Request mínimo
```json
{
  "cod_municipio": "05001",
  "anio": 2024,
  "semana_epi": 32
}
```

### Request completo (cuando el usuario edita parámetros)
```json
{
  "cod_municipio": "05001",
  "anio": 2024,
  "semana_epi": 32,
  "altitud_msnm": 1523.0,
  "cat_altitud": "Medio (1.000-1.800 m)",
  "poblacion": 2351077.0,
  "temp_media_c": 22.5,
  "humedad_pct": 78.0,
  "precip_mm": 45.3
}
```

### Response
```json
{
  "index": 0,
  "cod_municipio": "05001",
  "anio": 2024,
  "semana_epi": 32,
  "prediction": 1,
  "probability": 0.87,
  "label": "ALERTA"
}
```

### Documentación Swagger interactiva
```
http://<EC2-PUBLIC-IP>:8001/docs
```

---

## 🔧 Comandos útiles

```bash
# Ver contenedores
docker ps

# Logs en tiempo real
docker compose logs -f

# Reiniciar
docker compose restart

# Detener
docker compose down

# Reconstruir desde cero
docker compose down
docker compose up --build -d

# Ver uso de recursos
docker stats
```

---

## 📊 Modelo

| Métrica | Validation (2017-2019) | Test (2022-2024) |
|---|---|---|
| **F2-score** | ~0.78 | ~0.78 |
| **AUC-ROC** | ~0.87 | ~0.87 |
| **Recall** | ~0.85 | ~0.83 |
| **Precision** | ~0.55 | ~0.58 |

**Variable objetivo:** `alerta=1` si casos ≥ P75 histórico (municipio, semana)
**Métrica principal:** F2 (penaliza falsos negativos 2x más que falsos positivos)
**Justificación:** falsa alarma (fumigación preventiva) cuesta menos que brote no detectado

---

## 🛣️ Roadmap (trabajo futuro)

- [ ] Migrar feature engineering al lakehouse (Glue Spark)
- [ ] Persistir features en RDS PostgreSQL (tabla `gold.dengue_features`)
- [ ] Pipeline automático bronze → silver → gold con triggers S3
- [ ] Retraining automático con AWS Step Functions + MLflow Model Registry
- [ ] API Gateway con autenticación
- [ ] CloudWatch metrics y alertas
- [ ] CI/CD con GitHub Actions
