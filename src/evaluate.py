import json
import joblib
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, classification_report,
    confusion_matrix, precision_recall_curve,
)
from src.config import (MODELS_DIR, REPORTS_DIR, METRICS_PATH, CM_PATH,
                        MLFLOW_URI, EXPERIMENT_NAME, THRESHOLD_RECALL_MIN)
from src.preprocessing import load_splits


def find_threshold(y_true, y_proba, recall_min: float = THRESHOLD_RECALL_MIN) -> float:
    precs, recs, threshs = precision_recall_curve(y_true, y_proba)
    f1s    = 2 * precs * recs / (precs + recs + 1e-8)
    viable = recs[:-1] >= recall_min
    if viable.any():
        return float(threshs[np.argmax(np.where(viable, f1s[:-1], 0))])
    return float(threshs[np.argmax(f1s[:-1])])


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    prev   = float(y_true.mean())
    auc_pr = average_precision_score(y_true, y_proba)
    return {
        "auc_roc":          round(roc_auc_score(y_true, y_proba), 4),
        "auc_pr":           round(auc_pr, 4),
        "baseline_auc_pr":  round(prev, 4),
        "gain_vs_baseline": round(auc_pr - prev, 4),
        "gain_ratio":       round(auc_pr / prev, 2),
        "threshold":        round(threshold, 4),
        "recall":           round(recall_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_true, y_pred), 4),
        "TP":               int(cm[1, 1]),
        "FP":               int(cm[0, 1]),
        "FN":               int(cm[1, 0]),
        "TN":               int(cm[0, 0]),
    }


def save_confusion_matrix(y_true, y_proba, threshold: float):
    y_pred  = (y_proba >= threshold).astype(int)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=["Non réadmis", "Réadmis <30j"],
                yticklabels=["Non réadmis", "Réadmis <30j"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title(f"Confusion Matrix (seuil={threshold:.3f})")
    plt.tight_layout()
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_PATH, dpi=120)
    plt.close()
    print(f"[evaluate] Confusion matrix → {CM_PATH}")


def evaluate() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    _, X_test, _, y_test = load_splits()
    model = joblib.load(MODELS_DIR / "best_model.pkl")

    y_proba   = model.predict_proba(X_test)[:, 1]
    threshold = find_threshold(y_test, y_proba)
    metrics   = compute_metrics(y_test, y_proba, threshold)

    print("\n" + "=" * 50)
    print("  RÉSULTATS TEST SET")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    y_pred = (y_proba >= threshold).astype(int)
    print("\n" + classification_report(
        y_test, y_pred, target_names=["Non réadmis", "Réadmis <30j"]
    ))

    # Sauvegarde
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_test, y_proba, threshold)
    joblib.dump(threshold, MODELS_DIR / "threshold.pkl")

    with mlflow.start_run():
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(CM_PATH))

    print(f"[evaluate] Métriques → {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate()