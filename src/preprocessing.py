import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import (PROCESSED_DIR, MODELS_DIR, TEST_SIZE,
                        RANDOM_STATE)


def split(df: pd.DataFrame):
    TARGET = "readmitted"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[split] Train={X_train.shape}  Test={X_test.shape}")
    print(f"        Positifs train={y_train.mean():.3f} | test={y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_train, PROCESSED_DIR / "X_train.pkl")
    joblib.dump(X_test,  PROCESSED_DIR / "X_test.pkl")
    joblib.dump(y_train, PROCESSED_DIR / "y_train.pkl")
    joblib.dump(y_test,  PROCESSED_DIR / "y_test.pkl")
    print(f"[save_splits] → {PROCESSED_DIR}")


def load_splits():
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    X_test  = joblib.load(PROCESSED_DIR / "X_test.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")
    y_test  = joblib.load(PROCESSED_DIR / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def validate(X_train, X_test):
    assert not np.isnan(X_train.select_dtypes("number").values).any(), \
        "NaN dans X_train (colonnes numériques)"
    assert X_train.shape[1] == X_test.shape[1], \
        "Nombre de colonnes différent train/test"
    print(f"[validate] OK — {X_train.shape[1]} features, 0 NaN numérique")