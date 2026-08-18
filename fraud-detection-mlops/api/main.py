"""
FastAPI service for real-time fraud scoring.

Loads the trained model + feature engineer once at startup and exposes:
 - POST /predict        -> score a single transaction
 - POST /predict/batch   -> score a list of transactions
 - GET  /health          -> liveness + model metadata

Run locally:
    uvicorn api.main:app --reload --port 8000

Run with Docker: see Dockerfile / docker-compose.yml
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import FraudFeatureEngineer  # noqa: E402  (needed for joblib unpickling)

from api.schemas import HealthResponse, PredictionResponse, Transaction

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_feature_engineer = None
_feature_cols = None
_metrics = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_engineer, _feature_cols, _metrics
    try:
        _model = joblib.load(MODEL_DIR / "model.joblib")
        _feature_engineer = joblib.load(MODEL_DIR / "feature_engineer.joblib")
        _feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
        with open(MODEL_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Model artifacts not found. Run `python src/train.py` before starting the API."
        ) from e
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score(transactions: list[Transaction]) -> pd.DataFrame:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([t.model_dump() for t in transactions])
    df["is_fraud"] = 0  # placeholder, dropped inside transform
    X = _feature_engineer.transform(df)
    X = X.reindex(columns=_feature_cols, fill_value=0)

    probs = _model.predict_proba(X)[:, 1]
    threshold = _metrics.get("decision_threshold", 0.5)

    return pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "fraud_probability": probs.round(4),
        "is_fraud_flag": probs >= threshold,
        "decision_threshold": threshold,
    })


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_version="xgboost-v1",
        pr_auc=_metrics.get("pr_auc", 0.0),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    result = _score([txn]).iloc[0]
    return PredictionResponse(**result.to_dict())


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(transactions: list[Transaction]):
    if len(transactions) > 5000:
        raise HTTPException(status_code=400, detail="Batch size limited to 5000 transactions")
    result = _score(transactions)
    return [PredictionResponse(**row) for row in result.to_dict(orient="records")]


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
    }
