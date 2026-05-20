# Dashboard AI-lerta v3 — Doble Clasificación

Dashboard que consume la API REST `/predict/realtime` en EC2 separada.

## Arquitectura

```
┌────────────────────────┐         ┌────────────────────────┐
│  EC2 #1: Dashboard     │         │  EC2 #2: API           │
│  ┌──────────────────┐  │  HTTP   │  ┌──────────────────┐  │
│  │  Dash + gunicorn │──┼────────►│  │ FastAPI + XGBoost│  │
│  │  Puerto 8050     │  │         │  │  Puerto 8001     │  │
│  └──────────────────┘  │         │  └──────────────────┘  │
└────────────────────────┘         └────────────────────────┘
   54.x.x.x (nueva)                   54.226.141.151 (existente)
```

## Características v3

- ✅ Consume `POST /api/v1/predict/realtime` en EC2 separada
- ✅ Cascada departamento → municipio (1,121 municipios DIVIPOLA)
- ✅ VLOOKUP automático de clima 2025 al cambiar municipio/semana
- ✅ Doble clasificación:
  - **Zona INS** (Epidémica/Alerta/Éxito/Seguridad)
  - **Nivel de Urgencia** (ALTA/MEDIA/PRECAUCIÓN/NORMAL)
- ✅ Recomendaciones diferenciadas por nivel
- ✅ Mapa con probabilidad cruda

## Setup local (testing)

```bash
# 1. Ambiente virtual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generar catálogos
#    Coloca divipola.parquet en la raíz
python prepare_assets.py
#    → crea assets/divipola.csv

# 3. Copiar clima 2025
cp /ruta/clima_colombia_2025.csv assets/

# 4. Ejecutar
python app.py
```

Abrir: http://localhost:8050

## Despliegue en EC2 #2 (dashboard)

### 1. Crear EC2 nueva

| Setting | Valor |
|---|---|
| Name | `dengue-dashboard` |
| AMI | Ubuntu Server 22.04 LTS |
| Instance type | t3.small |
| Storage | 16 GiB gp3 |
| Key pair | usa la misma de la API o crea nueva |
| Security Group | nuevo: `dengue-dashboard-sg` |

**Reglas inbound del security group:**
- SSH (22) — My IP
- Custom TCP (8050) — Anywhere-IPv4 (público para usuarios)

### 2. Conectarse y preparar

```bash
ssh -i tu-key.pem ubuntu@<EC2-DASHBOARD-IP>

sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl start docker
sudo usermod -aG docker ubuntu
exit
ssh -i tu-key.pem ubuntu@<EC2-DASHBOARD-IP>
```

### 3. Subir el código desde tu Mac

```bash
scp -i tu-key.pem -r dashboard-v3 ubuntu@<EC2-DASHBOARD-IP>:/home/ubuntu/
```

### 4. En la EC2: levantar

```bash
cd /home/ubuntu/dashboard-v3
docker compose up -d --build
docker compose logs -f
```

Cuando veas el mensaje de gunicorn corriendo, abrir en navegador:
```
http://<EC2-DASHBOARD-IP>:8050
```

## Cambiar URL de la API

La URL de la API se pasa como variable de entorno. Edita `docker-compose.yml`:

```yaml
environment:
  - API_URL=http://otra-ip:8001
```

Luego:
```bash
docker compose down
docker compose up -d --build
```

## Comandos útiles

```bash
docker ps                          # ver contenedores
docker compose logs -f --tail=50   # logs en tiempo real
docker compose restart             # reiniciar
docker compose down                # detener
docker stats                       # uso de recursos
```

## Estructura

```
dashboard-v3/
├── app.py                     # Dash app
├── prepare_assets.py          # Convierte divipola.parquet → CSV
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── divipola.parquet           # input (no se sube a EC2 después)
└── assets/                    # se sube a EC2
    ├── divipola.csv
    └── clima_colombia_2025.csv
```
