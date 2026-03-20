from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import json

app    = FastAPI(title="Fraud Detection API", version="1.0")
model  = joblib.load("checkpoints/best_model.pkl")
scaler = joblib.load("checkpoints/scaler.pkl")
params = json.load(open("outputs/metriques.json"))

SEUIL     = params["seuil"]
explainer = shap.TreeExplainer(model)

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]


class Transaction(BaseModel):
    features: list  # 30 valeurs : V1-V28 + Amount_scaled + Time_scaled


@app.get("/")
def root():
    return {
        "model":  "XGBoost fraud detection",
        "auroc":  params["auroc_test"],
        "f1":     params["f1_test"],
        "seuil":  SEUIL
    }


@app.post("/predict")
def predict(transaction: Transaction):
    X = np.array(transaction.features).reshape(1, -1)

    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= SEUIL)

    # top 3 features SHAP pour expliquer la decision
    sv      = explainer.shap_values(X)[0]
    top3_idx = np.argsort(np.abs(sv))[-3:][::-1]
    top3    = [
        {"feature": FEATURE_NAMES[i], "shap_value": round(float(sv[i]), 4)}
        for i in top3_idx
    ]

    return {
        "fraude":             bool(label),
        "probabilite":        round(proba, 4),
        "top3_features_shap": top3,
        "seuil_utilise":      SEUIL
    }


# pour lancer : uvicorn src.api:app --reload
# exemple de requete :
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"features": [0.1, -1.2, 0.5, ...]}'