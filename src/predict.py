import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR
from src.data_loader import encode_inference, feature_engineering
from src.utils.logger import get_logger

logger = get_logger("src.predict")


def load_artifacts() -> dict:
    arts = {
        "model": joblib.load(MODELS_DIR / "best_model.pkl"),
        "features": joblib.load(MODELS_DIR / "feature_names.pkl"),
        "threshold": joblib.load(MODELS_DIR / "threshold.pkl"),
    }
    arts["explainer"] = shap.TreeExplainer(arts["model"])
    logger.info(f"Artefacts chargés — {len(arts['features'])} features")
    return arts


def _prepare(patient_data: dict, features: list) -> pd.DataFrame:
    """Prépare un patient pour l'inférence (sans colonne readmitted)."""
    df = pd.DataFrame([patient_data])

    # Supprimer readmitted si présent par erreur
    df = df.drop(columns=["readmitted"], errors="ignore")

    df = encode_inference(df)
    df = feature_engineering(df)

    # Aligner les colonnes avec ce que le modèle attend
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Supprimer toute colonne inattendue
    cols_to_keep = [c for c in features if c in df.columns]
    df = df[cols_to_keep]

    # Colonnes manquantes → 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


def _shap_top3(explainer, X_row: pd.DataFrame, feature_names: list) -> list:
    """Retourne les 3 features ayant le plus fort impact SHAP."""
    try:
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        return [
            {
                "feature": feature_names[i],
                "shap_value": round(float(sv[i]), 4),
                "direction": "↑ risque" if sv[i] > 0 else "↓ risque",
            }
            for i in top_idx
        ]
    except Exception as e:
        logger.warning(f"SHAP local failed : {e}")
        return []


def predict_single(patient_data: dict, artifacts: dict = None) -> dict:
    if artifacts is None:
        artifacts = load_artifacts()

    # Nettoyer les données d'entrée (supprimer readmitted si présent)
    data = {k: v for k, v in patient_data.items() if k != "readmitted"}

    df = _prepare(data, artifacts["features"])
    proba = float(artifacts["model"].predict_proba(df)[0, 1])
    pred = int(proba >= artifacts["threshold"])
    top3 = _shap_top3(artifacts["explainer"], df, artifacts["features"])

    risk = "Élevé" if proba >= 0.15 else "Modéré" if proba >= 0.098 else "Faible"

    result = {
        "prediction": pred,
        "probability": round(proba, 4),
        "risk_level": risk,
        "threshold": round(float(artifacts["threshold"]), 4),
        "label": "Réadmission probable (<30j)" if pred else "Pas de réadmission",
        "top_risk_factors": top3,
    }
    logger.info(f"Prédiction : {result['label']} | proba={proba:.3f}")
    return result


def predict_batch(df: pd.DataFrame, artifacts: dict = None) -> pd.DataFrame:
    if artifacts is None:
        artifacts = load_artifacts()

    df_copy = df.drop(columns=["readmitted"], errors="ignore").copy()
    df_enc = feature_engineering(encode_inference(df_copy))
    feats = artifacts["features"]

    for col in feats:
        if col not in df_enc.columns:
            df_enc[col] = 0

    probas = artifacts["model"].predict_proba(df_enc[feats])[:, 1]
    preds = (probas >= artifacts["threshold"]).astype(int)

    return pd.DataFrame(
        {
            "prediction": preds,
            "probability": probas.round(4),
            "risk_level": [
                "Élevé" if p >= 0.15 else "Modéré" if p >= 0.098 else "Faible"
                for p in probas
            ],
        }
    )
