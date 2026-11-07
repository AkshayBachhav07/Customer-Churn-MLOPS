import os
import pytest
from fastapi.testclient import TestClient

if not os.path.exists("models/model.pkl"):
    import train

from app import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["model_artifact_loaded"] is True

def test_model_artifact_exists():
    assert os.path.isfile("models/model.pkl")

def test_predict_churn_endpoint():
    payload = {
        "tenure_months": 2,
        "monthly_charges": 105.0,
        "contract_type": 0,
        "internet_service": 1,
        "payment_method": 0,
        "support_tickets_count": 4
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["High Risk (Will Churn)", "Low Risk (Loyal)"]
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_validation_failure():
    bad_payload = {
        "tenure_months": 2,
        "monthly_charges": 105.0,
        "contract_type": 10, # Must be 0, 1, or 2
        "internet_service": 1,
        "payment_method": 0,
        "support_tickets_count": 4
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
