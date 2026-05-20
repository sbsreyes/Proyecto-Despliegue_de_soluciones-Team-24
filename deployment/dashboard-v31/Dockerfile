FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8050

WORKDIR /app

# curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app
COPY app.py .
COPY assets/ ./assets/

EXPOSE 8050

# gunicorn en producción (2 workers es suficiente para el tráfico de demo)
CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
