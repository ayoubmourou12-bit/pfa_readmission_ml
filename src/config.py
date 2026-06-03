from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_params() -> dict:
    with open(ROOT / "params.yaml", "r") as f:
        return yaml.safe_load(f)


PARAMS = load_params()

# ── Chemins ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT / PARAMS["data"]["raw_path"]
PROCESSED_DIR = ROOT / PARAMS["data"]["processed_dir"]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "readmission_hospital"

# ── Data ───────────────────────────────────────────────────────────────────────
TEST_SIZE = PARAMS["data"]["test_size"]
RANDOM_STATE = PARAMS["data"]["random_state"]
DEDUP_COL = PARAMS["data"]["dedup_col"]
DROP_COLS = PARAMS["data"]["drop_cols"]
DROP_DISCH_ID = PARAMS["data"]["drop_discharge_id"]

# ── Preprocessing ──────────────────────────────────────────────────────────────
MED_COLS = PARAMS["preprocessing"]["med_cols"]
WINSORIZE_COLS = PARAMS["preprocessing"]["winsorize_cols"]
WINSORIZE_QUANTILE = PARAMS["preprocessing"]["winsorize_quantile"]

# ── Training ───────────────────────────────────────────────────────────────────
TRAINING = PARAMS["training"]
N_ITER = PARAMS["training"]["n_iter_search"]

# ── Evaluation ─────────────────────────────────────────────────────────────────
THRESHOLD_RECALL_MIN = PARAMS["evaluation"]["threshold_recall_min"]
METRICS_PATH = ROOT / PARAMS["evaluation"]["metrics_path"]
CM_PATH = ROOT / PARAMS["evaluation"]["confusion_matrix_path"]
MAIN_METRIC = PARAMS["evaluation"]["main_metric"]
