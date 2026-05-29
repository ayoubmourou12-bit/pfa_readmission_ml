from pydantic import BaseModel, Field, field_validator
from typing import Literal

VALID_AGE = {
    "[0-10)",  "[10-20)", "[20-30)", "[30-40)",
    "[40-50)", "[50-60)", "[60-70)", "[70-80)",
    "[80-90)", "[90-100)",
}
VALID_MED_STATUS = {"No", "Steady", "Up", "Down"}
VALID_GLUCOSE    = {"None", "Norm", ">200", ">300"}
VALID_A1C        = {"None", "Norm", ">7", ">8"}


class PatientInput(BaseModel):
    # ── Numériques obligatoires ───────────────────────────────────────────────
    time_in_hospital:         int = Field(..., ge=1,  le=14,
                                          description="Durée séjour (jours)")
    num_lab_procedures:       int = Field(..., ge=1,  le=132,
                                          description="Nb procédures lab")
    num_procedures:           int = Field(..., ge=0,  le=6,
                                          description="Nb procédures")
    num_medications:          int = Field(..., ge=1,  le=79,
                                          description="Nb médicaments")
    number_outpatient:        int = Field(..., ge=0,
                                          description="Visites ambulatoires antérieures")
    number_inpatient:         int = Field(..., ge=0,
                                          description="Hospitalisations antérieures")
    number_emergency:         int = Field(..., ge=0,
                                          description="Urgences antérieures")
    number_diagnoses:         int = Field(..., ge=1, le=16,
                                          description="Nb diagnostics")

    # ── Catégorielles obligatoires ────────────────────────────────────────────
    age:    str = Field(..., description="Tranche d'âge ex: [70-80)")
    gender: Literal["Male", "Female"]
    race:   str = Field(default="Caucasian")

    # ── IDs admission ─────────────────────────────────────────────────────────
    admission_type_id:        int = Field(default=1, ge=1)
    discharge_disposition_id: int = Field(default=1, ge=1)
    admission_source_id:      int = Field(default=7, ge=1)

    # ── Diagnostics ICD-9 ─────────────────────────────────────────────────────
    diag_1: str = Field(..., description="Diagnostic principal ICD-9")
    diag_2: str = Field(default="0")
    diag_3: str = Field(default="0")

    # ── Tests biologiques ─────────────────────────────────────────────────────
    max_glu_serum: str = Field(default="None")
    A1Cresult:     str = Field(default="None")

    # ── Médicaments et traitement ─────────────────────────────────────────────
    change:      Literal["Ch", "No"] = Field(default="No")
    diabetesMed: Literal["Yes", "No"] = Field(default="No")
    insulin:     str = Field(default="No")
    metformin:   str = Field(default="No")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v not in VALID_AGE:
            raise ValueError(f"age invalide '{v}'. Attendu : {sorted(VALID_AGE)}")
        return v

    @field_validator("max_glu_serum")
    @classmethod
    def validate_glucose(cls, v):
        if v not in VALID_GLUCOSE:
            raise ValueError(f"max_glu_serum invalide '{v}'. Attendu : {VALID_GLUCOSE}")
        return v

    @field_validator("A1Cresult")
    @classmethod
    def validate_a1c(cls, v):
        if v not in VALID_A1C:
            raise ValueError(f"A1Cresult invalide '{v}'. Attendu : {VALID_A1C}")
        return v

    @field_validator("insulin", "metformin")
    @classmethod
    def validate_med_status(cls, v):
        if v not in VALID_MED_STATUS:
            raise ValueError(f"Statut médicament invalide '{v}'."
                             f" Attendu : {VALID_MED_STATUS}")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "time_in_hospital": 5, "num_lab_procedures": 45,
            "num_procedures": 1,   "num_medications": 18,
            "number_outpatient": 0,"number_inpatient": 1,
            "number_emergency": 0, "number_diagnoses": 7,
            "age": "[70-80)",      "gender": "Male",
            "diag_1": "410",       "diag_2": "250",
            "diag_3": "401",       "A1Cresult": ">7",
            "change": "Ch",        "diabetesMed": "Yes",
            "insulin": "Up",
        }
    }}


class PredictionResponse(BaseModel):
    prediction:       int
    probability:      float
    risk_level:       str
    threshold:        float
    label:            str
    top_risk_factors: list[dict]  # SHAP local top 3