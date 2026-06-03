import json
import time

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import PatientInput, PredictionResponse
from src.config import METRICS_PATH, MODELS_DIR
from src.predict import predict_single
from src.utils.logger import get_logger

logger = get_logger("api.routes")
router = APIRouter()

# Artefacts chargés une seule fois au démarrage (injectés depuis app.py)
_artifacts: dict = {}


def set_artifacts(arts: dict):
    _artifacts.update(arts)


@router.get("/health", tags=["System"])
def health():
    logger.info("GET /health")
    return {
        "status": "ok",
        "model": "LightGBM — NaN natif (AUC-PR optimisé)",
        "version": "1.0.0",
        "model_loaded": bool(_artifacts),
    }


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    start = time.time()
    try:
        result = predict_single(patient.model_dump(), _artifacts)
        elapsed = round(time.time() - start, 3)
        logger.info(
            f"POST /predict | pred={result['prediction']} "
            f"proba={result['probability']:.3f} "
            f"risk={result['risk_level']} | {elapsed}s"
        )
        return result
    except Exception as e:
        logger.error(f"POST /predict ERROR : {e}")
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/model-info", tags=["System"])
def model_info():
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        features = joblib.load(MODELS_DIR / "feature_names.pkl")
        info = {
            "model_type": "LightGBM (NaN natif, scoring=AUC-PR)",
            "n_features": len(features),
            "metrics": metrics,
            "dataset": "UCI Diabetes 130-US (~68k patients)",
            "target": "Réadmission dans les 30 jours",
            "main_metric": "AUC-PR",
            "prevalence": metrics.get("baseline_auc_pr", "N/A"),
            "auc_pr": metrics.get("auc_pr", "N/A"),
            "auc_roc": metrics.get("auc_roc", "N/A"),
        }
        logger.info("GET /model-info")
        return info
    except Exception as e:
        logger.error(f"GET /model-info ERROR : {e}")
        raise HTTPException(status_code=500, detail=str(e))
