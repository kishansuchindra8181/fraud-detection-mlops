
from pathlib import Path
from math import isfinite

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names = artifact["feature_names"]
threshold = artifact["threshold"]

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict potentially fraudulent credit card transactions.",
    version="1.0.0",
)


class TransactionInput(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: TransactionInput):
    supplied = data.features

    missing = set(feature_names) - set(supplied)
    extra = set(supplied) - set(feature_names)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra),
            },
        )

    if not all(isfinite(value) for value in supplied.values()):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    features = pd.DataFrame(
        [[supplied[name] for name in feature_names]],
        columns=feature_names,
    )

    fraud_probability = float(model.predict_proba(features)[0][1])
    prediction = int(fraud_probability >= threshold)

    return {
        "prediction": prediction,
        "label": (
            "Fraudulent transaction"
            if prediction == 1
            else "Legitimate transaction"
        ),
        "fraud_probability": round(fraud_probability, 6),
        "threshold": threshold,
    }
