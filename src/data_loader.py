import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (RAW_DATA_PATH, DROP_COLS, DROP_DISCH_ID,
                        DEDUP_COL, MED_COLS, WINSORIZE_COLS, WINSORIZE_QUANTILE)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    df = pd.read_csv(path)
    required = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_inpatient",
                "number_emergency", "age", "diag_1", "diag_2", "diag_3",
                "change", "diabetesMed", "readmitted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    print(f"[load] {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # Supprimer colonnes inutiles
    cols_drop = [c for c in DROP_COLS if c in df.columns
                 and c not in [DEDUP_COL, "encounter_id"]]
    df = df.drop(columns=cols_drop, errors="ignore")

    # Supprimer décédés
    df = df[df["discharge_disposition_id"] != DROP_DISCH_ID]

    # Supprimer lignes invalides
    drop_idx = set()
    for col in ["race"]:
        if col in df.columns:
            drop_idx.update(df[df[col] == "?"].index)
    if "gender" in df.columns:
        drop_idx.update(df[df["gender"] == "Unknown/Invalid"].index)
    df = df.drop(index=drop_idx)

    # Remplacer '?' par NaN dans diag (gérés nativement par LightGBM)
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].replace("?", np.nan)

    # Dédupliquer : garder 1ère visite par patient
    if DEDUP_COL in df.columns:
        df = df.drop_duplicates(subset=[DEDUP_COL], keep="first")

    # Supprimer IDs
    df = df.drop(columns=["encounter_id", DEDUP_COL], errors="ignore")
    df = df.reset_index(drop=True)

    print(f"[clean] {n0:,} → {len(df):,} lignes ({n0-len(df):,} supprimées)")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    # Target
    df["readmitted"] = df["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})

    # Binaires
    df["change"]      = df["change"].map({"Ch": 1, "No": 0})
    df["gender"]      = df["gender"].map({"Male": 1, "Female": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    # Age ordinal
    age_map = {f"[{10*i}-{10*(i+1)})": i+1 for i in range(10)}
    df["age"] = df["age"].map(age_map)

    # Médicaments
    for col in [c for c in MED_COLS if c in df.columns]:
        df[col] = df[col].map({"No": 0, "Steady": 1, "Up": 1, "Down": 1})

    # Tests biologiques → flags binaires (94%/83% NaN)
    for src, tested, high, pos_vals in [
        ("max_glu_serum", "glu_tested", "glu_high",  [">200", ">300", 1]),
        ("A1Cresult",     "A1C_tested", "A1C_high",  [">7",   ">8",   1]),
    ]:
        if src in df.columns:
            df[tested] = df[src].notna().astype(int)
            df[high]   = df[src].isin(pos_vals).astype(int)
            df.drop(columns=[src], inplace=True)

    # Admission IDs → regroupement
    df["admission_type_id"] = df["admission_type_id"].replace(
        {2:1, 7:1, 6:5, 8:5})
    df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(
        {6:1,8:1,9:1,13:1, 3:2,4:2,5:2,14:2,22:2,23:2,24:2,
         12:10,15:10,16:10,17:10, 25:18,26:18})
    df["admission_source_id"] = df["admission_source_id"].replace(
        {2:1,3:1, 5:4,6:4,10:4,22:4,25:4,
         15:9,17:9,20:9,21:9, 13:11,14:11})

    # Race → OHE
    if "race" in df.columns:
        race_d = pd.get_dummies(df["race"], prefix="race", drop_first=True)
        df = pd.concat([df.drop(columns=["race"]), race_d], axis=1)

    print(f"[encode] Shape : {df.shape}")
    return df


def _parse_diag(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.where(~s.str.startswith("V", na=False), "0")
    s = s.where(~s.str.startswith("E", na=False), "0")
    s = s.replace("nan", np.nan)
    return pd.to_numeric(s, errors="coerce")


def _map_level1(s: pd.Series) -> pd.Series:
    f = s.apply(np.floor)
    r = pd.Series(0, index=s.index)
    r[((s>=390)&(s<460))|(f==785)] = 1
    r[((s>=460)&(s<520))|(f==786)] = 2
    r[((s>=520)&(s<580))|(f==787)] = 3
    r[f==250]                       = 4
    r[(s>=800)&(s<1000)]            = 5
    r[(s>=710)&(s<740)]             = 6
    r[((s>=580)&(s<630))|(f==788)] = 7
    r[(s>=140)&(s<240)]             = 8
    return r.astype(int)


def _map_level2(s: pd.Series) -> pd.Series:
    f = s.apply(np.floor)
    r = pd.Series(0, index=s.index)
    mapping = [
        (1,  (s>=390)&(s<399)),
        (2,  (s>=401)&(s<415)),
        (3,  (s>=415)&(s<460)),
        (4,  f==785),
        (5,  (s>=460)&(s<489)),
        (6,  (s>=490)&(s<497)),
        (7,  (s>=500)&(s<520)),
        (8,  f==786),
        (9,  (s>=520)&(s<530)),
        (10, (s>=530)&(s<544)),
        (11, (s>=550)&(s<554)),
        (12, (s>=555)&(s<580)),
        (13, f==787),
        (14, f==250),
        (15, (s>=800)&(s<1000)),
        (16, (s>=710)&(s<740)),
        (17, (s>=580)&(s<630)),
        (18, f==788),
        (19, (s>=140)&(s<240)),
        (20, (s>=240)&(s<280)&(f!=250)),
        (21, ((s>=680)&(s<710))|(f==782)),
        (22, (s>=290)&(s<320)),
    ]
    for val, mask in mapping:
        r[mask] = val
    return r.astype(int)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # ICD-9 vectorisé
    for suffix in ["1", "2", "3"]:
        col = f"diag_{suffix}"
        if col in df.columns:
            numeric = _parse_diag(df[col])
            df[f"level1_diag{suffix}"] = _map_level1(numeric.fillna(0))
            df[f"level2_diag{suffix}"] = _map_level2(numeric.fillna(0))
            df[f"diag{suffix}_missing"] = numeric.isna().astype(int)
            df.drop(columns=[col], inplace=True)

    # Winsorisation
    for col in [c for c in WINSORIZE_COLS if c in df.columns]:
        cap = df[col].quantile(WINSORIZE_QUANTILE)
        df[col] = df[col].clip(upper=cap)

    # Features dérivées
    df["service_utilization"] = (df.get("number_outpatient", 0) +
                                  df.get("number_emergency", 0) +
                                  df.get("number_inpatient", 0))
    df["care_intensity"]  = (df.get("num_lab_procedures", 0) +
                              df.get("num_procedures", 0) +
                              df.get("num_medications", 0))
    df["med_per_day"]     = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    df["lab_per_day"]     = df.get("num_lab_procedures", pd.Series(0, index=df.index)) / \
                            df["time_in_hospital"].clip(lower=1)
    df["had_inpatient"]   = (df.get("number_inpatient", 0) > 0).astype(int)
    df["had_emergency"]   = (df.get("number_emergency", 0) > 0).astype(int)
    df["had_outpatient"]  = (df.get("number_outpatient", 0) > 0).astype(int)
    df["polypharmacy"]    = (df["num_medications"] >= 15).astype(int)
    df["extreme_poly"]    = (df["num_medications"] >= 25).astype(int)
    df["long_stay"]       = (df["time_in_hospital"] > 7).astype(int)
    df["very_short_stay"] = (df["time_in_hospital"] <= 2).astype(int)
    df["age_x_inpatient"] = df["age"] * df.get("number_inpatient", 0)
    df["age_x_emergency"] = df["age"] * df.get("number_emergency", 0)
    df["age_x_meds"]      = df["age"] * df["num_medications"]
    df["n_serious_diag"]  = (
        df["level1_diag1"].isin([1,2,4]).astype(int) +
        df["level1_diag2"].isin([1,2,4]).astype(int) +
        df["level1_diag3"].isin([1,2,4]).astype(int)
    )
    df["uncontrolled_dm"] = (
        (df.get("A1C_high", 0) == 1) &
        (df.get("diabetesMed", 0) == 1) &
        (df.get("change", 0) == 1)
    ).astype(int)
    med_present = [c for c in MED_COLS if c in df.columns]
    df["numchange"] = df[med_present].sum(axis=1)
    if "insulin" in df.columns:
        df["insulin_change"] = ((df["insulin"] == 1) &
                                (df.get("change", 0) == 1)).astype(int)

    
    if "readmitted" in df.columns:
        df = df.dropna(subset=["readmitted"])
    print(f"[feature_engineering] Shape : {df.shape}")
    return df


def build_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df = clean(df)
    df = encode(df)
    df = feature_engineering(df)
    return df

# Ajouter à la fin de src/data_loader.py

def encode_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodage pour l'inférence (un seul patient).
    Identique à encode() mais sans la colonne 'readmitted'.
    """
    # Binaires
    if "change" in df.columns:
        df["change"]      = df["change"].map({"Ch": 1, "No": 0})
    if "gender" in df.columns:
        df["gender"]      = df["gender"].map({"Male": 1, "Female": 0})
    if "diabetesMed" in df.columns:
        df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    # Age ordinal
    if "age" in df.columns:
        age_map = {f"[{10*i}-{10*(i+1)})": i+1 for i in range(10)}
        df["age"] = df["age"].map(age_map)

    # Médicaments
    for col in [c for c in MED_COLS if c in df.columns]:
        df[col] = df[col].map({"No": 0, "Steady": 1, "Up": 1, "Down": 1})

    # Tests biologiques → flags
    for src, tested, high, pos_vals in [
        ("max_glu_serum", "glu_tested", "glu_high",  [">200", ">300", 1]),
        ("A1Cresult",     "A1C_tested", "A1C_high",  [">7",   ">8",   1]),
    ]:
        if src in df.columns:
            df[tested] = df[src].notna().astype(int)
            df[high]   = df[src].isin(pos_vals).astype(int)
            df.drop(columns=[src], inplace=True)

    # IDs admission
    if "admission_type_id" in df.columns:
        df["admission_type_id"] = df["admission_type_id"].replace(
            {2:1, 7:1, 6:5, 8:5})
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(
            {6:1,8:1,9:1,13:1, 3:2,4:2,5:2,14:2,22:2,23:2,24:2,
             12:10,15:10,16:10,17:10, 25:18,26:18})
    if "admission_source_id" in df.columns:
        df["admission_source_id"] = df["admission_source_id"].replace(
            {2:1,3:1, 5:4,6:4,10:4,22:4,25:4,
             15:9,17:9,20:9,21:9, 13:11,14:11})

    # Race OHE
    if "race" in df.columns:
        race_d = pd.get_dummies(df["race"], prefix="race", drop_first=True)
        df = pd.concat([df.drop(columns=["race"]), race_d], axis=1)

    return df