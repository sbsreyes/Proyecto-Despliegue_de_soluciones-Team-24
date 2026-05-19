# ============================================================
# Dockerfile — AI-lerta Dengue API
# ============================================================
#
# Construir y correr:
#   docker-compose up --build
#
# La API queda disponible en http://localhost:8001/docs
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8001

WORKDIR /app

# Dependencias del sistema (curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Dependencias de la API
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar el paquete del modelo desde el .whl
COPY packages/ packages/
RUN pip install --no-cache-dir packages/*.whl

# Código de la app
COPY app/ app/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
