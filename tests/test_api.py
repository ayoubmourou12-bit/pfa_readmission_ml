import pytest
from fastapi.testclient import TestClient

SAMPLE = {
    "time_in_hospital":          5,
    "num_lab_procedures":       45,
    "num_procedures":            1,
    "num_medications":          18,
    "number_outpatient":         0,
    "number_inpatient":          1,
    "number_emergency":          0,
    "number_diagnoses":          7,
    "age":                 "[70-80)",
    "gender":              "Male",
    "race":                "Caucasian",
    "admission_type_id":         1,
    "discharge_disposition_id":  1,
    "admission_source_id":       7,
    "diag_1":              "410",
    "diag_2":              "250",
    "diag_3":              "401",
    "max_glu_serum":       "None",
    "A1Cresult":           ">7",
    "change":              "Ch",
    "diabetesMed":         "Yes",
    "insulin":             "Up",
    "metformin":           "Steady",
}


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, f"Erreur : {r.json()}"
    d = r.json()
    assert "prediction"       in d
    assert "probability"      in d
    assert "risk_level"       in d
    assert "threshold"        in d
    assert "label"            in d
    assert "top_risk_factors" in d
    assert d["prediction"]  in [0, 1]
    assert 0.0 <= d["probability"] <= 1.0
    assert d["risk_level"] in ["Faible", "Modéré", "Élevé"]


def test_predict_missing_field_422(client):
    bad = SAMPLE.copy()
    del bad["diag_1"]           # champ obligatoire
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_invalid_age(client):
    bad = {**SAMPLE, "age": "[999-1000)"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_invalid_glucose(client):
    bad = {**SAMPLE, "max_glu_serum": "invalide"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    d = r.json()
    assert "metrics"   in d
    assert "auc_pr"    in d["metrics"]
    assert "auc_roc"   in d["metrics"]
    assert "n_features" in d


def test_predict_response_consistency(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, f"Erreur inattendue : {r.json()}"
    d = r.json()
    if d["prediction"] == 1:
        assert "Réadmission" in d["label"]
    else:
        assert "Pas" in d["label"]


def test_predict_shap_top3(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    d = r.json()
    top3 = d.get("top_risk_factors", [])
    # SHAP peut échouer gracieusement → liste vide ou 3 éléments
    assert isinstance(top3, list)
    if top3:
        assert len(top3) <= 3
        assert "feature"    in top3[0]
        assert "shap_value" in top3[0]
        assert "direction"  in top3[0]