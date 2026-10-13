# API image for the RiskRelay scoring service. Trained model artifacts in
# models/ are baked in (they're committed to the repo), so the container can
# serve immediately. Redis is a separate service (see docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

# system deps: libgomp is required by xgboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Redis backend by default in the container; REDIS_URL is injected by compose.
ENV FEATURE_STORE_BACKEND=redis \
    REDIS_URL=redis://redis:6379/0

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
