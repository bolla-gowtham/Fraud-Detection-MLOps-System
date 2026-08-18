# --- build stage: train the model into the image (fine for a demo/portfolio image) ---
FROM python:3.11-slim AS base

WORKDIR /app

# system deps for xgboost / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate data + train model at build time so the image is self-contained
RUN python data/generate_data.py --n_rows 250000 --out data/transactions.csv \
    && cd src && python train.py --data ../data/transactions.csv --out_dir ../models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
