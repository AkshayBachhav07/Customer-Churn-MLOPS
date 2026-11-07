import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End MLOps API for identifying subscriber churn risk.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model artifact not found. Execute train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=1, le=100, description="Months as subscriber", example=14)
    monthly_charges: float = Field(..., ge=10.0, le=250.0, description="Monthly billing amount", example=85.5)
    contract_type: int = Field(..., ge=0, le=2, description="0: Month-to-month, 1: 1-Year, 2: 2-Year", example=0)
    internet_service: int = Field(..., ge=0, le=2, description="0: DSL, 1: Fiber, 2: None", example=1)
    payment_method: int = Field(..., ge=0, le=1, description="0: Manual, 1: Auto-pay", example=0)
    support_tickets_count: int = Field(..., ge=0, le=20, description="Support tickets raised", example=3)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_artifact_loaded": model_loaded
    }

@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    model = get_model()
    input_df = pd.DataFrame([payload.model_dump()])

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    return {
        "churn_prediction": pred,
        "risk_level": "High Risk (Will Churn)" if pred == 1 else "Low Risk (Loyal)",
        "churn_probability": round(prob, 4),
        "status": "Success"
    }
