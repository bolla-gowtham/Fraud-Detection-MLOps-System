import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.main import app  # noqa: E402

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_TXN = {
    "transaction_id": 1,
    "customer_id": 100,
    "amount": 50.0,
    "hour_of_day": 14,
    "day_of_week": 1,
    "seconds_elapsed": 1000,
    "merchant_category": "grocery",
    "distance_from_home_km": 2.0,
    "is_online": 0,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "pr_auc" in body


def test_predict_single(client):
    resp = client.post("/predict", json=VALID_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud_flag"], bool)


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[VALID_TXN, {**VALID_TXN, "transaction_id": 2}])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2


def test_predict_invalid_amount_rejected(client):
    bad_txn = {**VALID_TXN, "amount": -10}
    resp = client.post("/predict", json=bad_txn)
    assert resp.status_code == 422


def test_predict_batch_too_large_rejected(client):
    batch = [{**VALID_TXN, "transaction_id": i} for i in range(5001)]
    resp = client.post("/predict/batch", json=batch)
    assert resp.status_code == 400
