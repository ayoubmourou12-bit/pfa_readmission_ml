"""
Détection de Data Drift et Concept Drift avec Evidently AI.
Compare les données d'entraînement avec les données de production.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    ColumnSummaryMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
)

from src.config import PROCESSED_DIR, REPORTS_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("src.drift")

# Features numériques surveillées en priorité
NUMERICAL_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_inpatient",
    "number_emergency", "number_diagnoses", "age",
    "care_intensity", "med_per_day", "service_utilization",
    "age_x_inpatient", "age_x_meds", "numchange",
]

CATEGORICAL_FEATURES = [
    "gender", "diabetesMed", "change", "insulin",
    "level1_diag1", "level1_diag2", "level1_diag3",
    "glu_tested", "glu_high", "A1C_tested", "A1C_high",
]

TARGET = "readmitted"
PREDICTION_COL = "prediction"


def load_reference_data(n_sample: int = 5000) -> pd.DataFrame:
    """Charge un échantillon des données d'entraînement comme référence."""
    import joblib
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")

    ref = X_train.copy()
    ref[TARGET] = y_train.values

    # Ajouter prédictions du modèle sur les données de référence
    model     = joblib.load(MODELS_DIR / "best_model.pkl")
    threshold = joblib.load(MODELS_DIR / "threshold.pkl")
    features  = joblib.load(MODELS_DIR / "feature_names.pkl")

    cols_avail = [c for c in features if c in ref.columns]
    probas     = model.predict_proba(ref[cols_avail])[:, 1]
    ref[PREDICTION_COL] = (probas >= threshold).astype(int)

    return ref.sample(n=min(n_sample, len(ref)), random_state=42)


def load_production_data(prod_path: Path = None) -> pd.DataFrame:
    """
    Charge les données de production.
    En l'absence de vraies données prod, simule un drift sur les données test.
    """
    import joblib
    from src.data_loader import build_dataset

    if prod_path and prod_path.exists():
        logger.info(f"Chargement données production : {prod_path}")
        df = pd.read_csv(prod_path)
    else:
        logger.warning("Pas de données prod réelles — simulation de drift sur X_test")
        X_test = joblib.load(PROCESSED_DIR / "X_test.pkl")
        y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")
        df = X_test.copy()
        df[TARGET] = y_test.values

        # Simuler un drift : augmenter l'âge et le nombre de médicaments
        if "age" in df.columns:
            noise = np.random.normal(loc=1.5, scale=0.5, size=len(df))
            df["age"] = (df["age"] + noise).clip(1, 10)
        if "num_medications" in df.columns:
            df["num_medications"] = (df["num_medications"] * 1.2).astype(int)
        if "number_inpatient" in df.columns:
            df["number_inpatient"] = (df["number_inpatient"] * 1.3).astype(int)

    # Ajouter prédictions modèle sur prod
    model     = joblib.load(MODELS_DIR / "best_model.pkl")
    threshold = joblib.load(MODELS_DIR / "threshold.pkl")
    features  = joblib.load(MODELS_DIR / "feature_names.pkl")

    cols_avail = [c for c in features if c in df.columns]
    probas     = model.predict_proba(df[cols_avail])[:, 1]
    df[PREDICTION_COL] = (probas >= threshold).astype(int)

    return df.sample(n=min(5000, len(df)), random_state=42)


def run_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    output_path: Path = None,
) -> dict:
    """
    Génère le rapport de drift complet (HTML + JSON).
    Retourne un dict résumé avec les métriques clés.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "drift"
    output_path.mkdir(parents=True, exist_ok=True)

    # Filtrer les colonnes disponibles dans les deux datasets
    num_feats = [c for c in NUMERICAL_FEATURES
                 if c in reference.columns and c in production.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES
                 if c in reference.columns and c in production.columns]

    column_mapping = ColumnMapping(
        target=TARGET if TARGET in reference.columns else None,
        prediction=PREDICTION_COL,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )

    # ── Rapport Data Drift ────────────────────────────────────────────────────
    drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name="age"),
        ColumnDriftMetric(column_name="num_medications"),
        ColumnDriftMetric(column_name="number_inpatient"),
        ColumnDriftMetric(column_name="time_in_hospital"),
        ColumnSummaryMetric(column_name="age"),
        ColumnSummaryMetric(column_name="num_medications"),
    ])

    drift_report.run(
        reference_data=reference,
        current_data=production,
        column_mapping=column_mapping,
    )

    # ── Test Suite (pass/fail) ────────────────────────────────────────────────
    test_suite = TestSuite(tests=[
        TestNumberOfDriftedColumns(lte=5),      # max 5 colonnes driftées
        TestShareOfDriftedColumns(lte=0.3),     # max 30% des colonnes
        TestColumnDrift(column_name="age"),
        TestColumnDrift(column_name="num_medications"),
        TestColumnDrift(column_name="number_inpatient"),
    ])

    test_suite.run(
        reference_data=reference,
        current_data=production,
        column_mapping=column_mapping,
    )

    # ── Rapport modèle (concept drift) ───────────────────────────────────────
    model_report = Report(metrics=[
        ClassificationQualityMetric(),
    ])

    if TARGET in reference.columns and TARGET in production.columns:
        model_report.run(
            reference_data=reference,
            current_data=production,
            column_mapping=column_mapping,
        )

    # ── Sauvegarder les rapports HTML ─────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_html = output_path / f"drift_report_{ts}.html"
    test_html  = output_path / f"test_suite_{ts}.html"

    drift_report.save_html(str(drift_html))
    test_suite.save_html(str(test_html))
    logger.info(f"Rapports HTML sauvegardés → {output_path}")

    # ── Extraire résumé JSON ──────────────────────────────────────────────────
    drift_json  = json.loads(drift_report.json())
    test_json   = json.loads(test_suite.json())

    dataset_drift = drift_json["metrics"][0]["result"]
    n_drifted     = dataset_drift.get("number_of_drifted_columns", 0)
    share_drifted = dataset_drift.get("share_of_drifted_columns", 0.0)
    drift_detected = dataset_drift.get("dataset_drift", False)

    tests_passed = sum(1 for t in test_json["tests"] if t["status"] == "SUCCESS")
    tests_failed = sum(1 for t in test_json["tests"] if t["status"] == "FAIL")

    summary = {
        "timestamp":          ts,
        "drift_detected":     drift_detected,
        "n_drifted_columns":  n_drifted,
        "share_drifted":      round(share_drifted, 3),
        "tests_passed":       tests_passed,
        "tests_failed":       tests_failed,
        "alert":              drift_detected or tests_failed > 0,
        "drift_report_html":  str(drift_html),
        "test_suite_html":    str(test_html),
    }

    # Sauvegarder le résumé JSON
    summary_path = output_path / f"drift_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Sauvegarder aussi le dernier résumé (pour le monitoring)
    with open(REPORTS_DIR / "drift_latest.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log_summary(summary)
    return summary


def _log_summary(summary: dict):
    logger.info("=" * 55)
    logger.info("  RAPPORT DE DRIFT")
    logger.info("=" * 55)
    status = "🔴 DRIFT DÉTECTÉ" if summary["drift_detected"] else "🟢 PAS DE DRIFT"
    logger.info(f"  Statut          : {status}")
    logger.info(f"  Colonnes driftées : {summary['n_drifted_columns']}"
                f" ({summary['share_drifted']:.1%})")
    logger.info(f"  Tests passés    : {summary['tests_passed']}")
    logger.info(f"  Tests échoués   : {summary['tests_failed']}")
    if summary["alert"]:
        logger.warning("⚠️  ALERTE : réentraînement recommandé !")
    logger.info("=" * 55)


if __name__ == "__main__":
    import sys

    prod_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print("Chargement des données de référence (train)...")
    ref  = load_reference_data()

    print("Chargement des données de production...")
    prod = load_production_data(prod_path)

    print(f"Référence : {ref.shape} | Production : {prod.shape}")
    print("Génération du rapport de drift...")

    summary = run_drift_report(ref, prod)

    print(f"\n{'='*55}")
    print(f"  Drift détecté : {summary['drift_detected']}")
    print(f"  Colonnes : {summary['n_drifted_columns']} driftées")
    print(f"  Tests : {summary['tests_passed']} OK / {summary['tests_failed']} KO")
    print(f"  Rapport : {summary['drift_report_html']}")
    print(f"{'='*55}")