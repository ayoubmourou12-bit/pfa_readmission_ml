import pytest
import pandas as pd
import numpy as np
from src.data_loader import clean, encode, feature_engineering


def _make_df(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        "encounter_id":             range(n),
        "patient_nbr":              range(n),
        "race":                     np.random.choice(["Caucasian","AfricanAmerican"], n),
        "gender":                   np.random.choice(["Male","Female"], n),
        "age":                      np.random.choice(["[50-60)","[60-70)","[70-80)"], n),
        "time_in_hospital":         np.random.randint(1, 14, n),
        "num_lab_procedures":       np.random.randint(1, 100, n),
        "num_procedures":           np.random.randint(0, 6, n),
        "num_medications":          np.random.randint(1, 40, n),
        "number_outpatient":        np.random.randint(0, 5, n),
        "number_inpatient":         np.random.randint(0, 5, n),
        "number_emergency":         np.random.randint(0, 3, n),
        "number_diagnoses":         np.random.randint(1, 9, n),
        "admission_type_id":        np.random.choice([1,2,3], n),
        "discharge_disposition_id": np.random.choice([1,2,3], n),
        "admission_source_id":      np.random.choice([1,7], n),
        "diag_1":                   np.random.choice(["250","410","V10","?"], n),
        "diag_2":                   np.random.choice(["250","410","?"], n),
        "diag_3":                   np.random.choice(["250","410"], n),
        "max_glu_serum":            np.random.choice([">200","Norm","None"], n),
        "A1Cresult":                np.random.choice([">7","Norm","None"], n),
        "change":                   np.random.choice(["Ch","No"], n),
        "diabetesMed":              np.random.choice(["Yes","No"], n),
        "insulin":                  np.random.choice(["No","Steady","Up"], n),
        "metformin":                np.random.choice(["No","Steady"], n),
        "repaglinide":              ["No"] * n,
        "nateglinide":              ["No"] * n,
        "chlorpropamide":           ["No"] * n,
        "glimepiride":              ["No"] * n,
        "glipizide":                ["No"] * n,
        "glyburide":                ["No"] * n,
        "pioglitazone":             ["No"] * n,
        "rosiglitazone":            ["No"] * n,
        "acarbose":                 ["No"] * n,
        "miglitol":                 ["No"] * n,
        "glyburide-metformin":      ["No"] * n,
        "tolazamide":               ["No"] * n,
        "metformin-pioglitazone":   ["No"] * n,
        "metformin-rosiglitazone":  ["No"] * n,
        "glimepiride-pioglitazone": ["No"] * n,
        "glipizide-metformin":      ["No"] * n,
        "troglitazone":             ["No"] * n,
        "tolbutamide":              ["No"] * n,
        "acetohexamide":            ["No"] * n,
        "readmitted":               np.random.choice(["<30",">30","NO"], n),
    })


def test_encode_target_binary():
    df = encode(_make_df())
    assert df["readmitted"].isin([0, 1]).all()


def test_encode_gender_binary():
    df = encode(_make_df())
    assert df["gender"].isin([0, 1]).all()


def test_encode_age_ordinal():
    df = encode(_make_df())
    assert df["age"].between(1, 10).all()


def test_feature_engineering_creates_cols():
    df = encode(_make_df())
    df = feature_engineering(df)
    expected = ["care_intensity", "med_per_day", "had_inpatient",
                "polypharmacy", "long_stay", "very_short_stay",
                "age_x_inpatient", "numchange", "n_serious_diag",
                "level1_diag1", "level2_diag1"]
    for col in expected:
        assert col in df.columns, f"Colonne manquante : {col}"


def test_feature_engineering_no_inf():
    df = encode(_make_df())
    df = feature_engineering(df)
    num = df.select_dtypes("number")
    assert not np.isinf(num.values).any(), "Valeurs infinies détectées"


def test_diag_missing_flags():
    df = _make_df()
    df = encode(df)
    df = feature_engineering(df)
    assert "diag1_missing" in df.columns
    assert df["diag1_missing"].isin([0, 1]).all()