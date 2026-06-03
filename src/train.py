import json
import time

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score)

from src.config import (EXPERIMENT_NAME, MAIN_METRIC, MLFLOW_URI, MODELS_DIR,
                        N_ITER, RANDOM_STATE, TRAINING)
from src.preprocessing import load_splits


def build_param_dist() -> dict:
    return {
        "n_estimators": [300, 500, 700, 1000],
        "max_depth": [4, 5, 6, -1],
        "num_leaves": [20, 31, 50, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_samples": [10, 20, 30, 50],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.1, 1.0, 5.0],
        "class_weight": ["balanced"],
    }


def train() -> lgb.LGBMClassifier:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_splits()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    base = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    search = RandomizedSearchCV(
        base,
        param_distributions=build_param_dist(),
        n_iter=N_ITER,
        cv=skf,
        scoring=MAIN_METRIC,  # average_precision (AUC-PR)
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )

    print("[train] Recherche hyperparamètres (scoring=AUC-PR)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[train] Terminé en {elapsed:.0f}s")
    print(f"[train] Meilleurs params : {search.best_params_}")
    print(f"[train] CV AUC-PR best  : {search.best_score_:.4f}")

    model = search.best_estimator_

    # CV AUC-ROC pour comparaison
    cv_auc = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
    )

    with mlflow.start_run():
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_auc_pr_best", round(search.best_score_, 4))
        mlflow.log_metric("cv_auc_roc_mean", round(float(cv_auc.mean()), 4))
        mlflow.log_metric("cv_auc_roc_std", round(float(cv_auc.std()), 4))
        mlflow.log_metric("train_time_s", round(elapsed, 1))
        mlflow.sklearn.log_model(model, "lightgbm_model")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_model.pkl")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_names.pkl")
    print(f"[train] Modèle sauvegardé → {MODELS_DIR / 'best_model.pkl'}")
    return model


if __name__ == "__main__":
    train()
