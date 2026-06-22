
# Prédiction de Réadmission Hospitalière — Pipeline MLOps

> Système d'aide à la décision médicale prédit le risque de réadmission
> hospitalière dans les 30 jours pour patients diabétiques.
> Pipeline MLOps complet : DVC · MLflow · FastAPI · Streamlit · Docker · CI/CD

---

## Table des matières

- [Contexte](#contexte)
- [Dataset](#dataset)
- [Résultats du modèle](#résultats-du-modèle)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Lancer le pipeline DVC](#lancer-le-pipeline-dvc)
- [MLflow — Tracking des expériences](#mlflow--tracking-des-expériences)
- [API REST](#api-rest)
- [Interface Streamlit](#interface-streamlit)
- [Tests automatiques](#tests-automatiques)
- [Docker](#docker)
- [Structure du dépôt](#structure-du-dépôt)

---

## Contexte

La réadmission hospitalière non planifiée dans les 30 jours suivant
une sortie représente un indicateur clé de la qualité des soins.
Aux États-Unis, les réadmissions évitables génèrent **26 milliards
de dollars de dépenses annuelles**.

Ce projet construit un pipeline MLOps complet qui :

- Prédit le risque de réadmission à partir de données cliniques tabulaires
- Explique chaque prédiction via SHAP (facteurs de risque individuels)
- Expose le modèle via une API REST documentée
- Fournit une interface médicale Streamlit accessible sans compétence technique
- Garantit la reproductibilité via DVC et la traçabilité via MLflow
- Automatise les tests et le déploiement via GitHub Actions

---

## Dataset

**Source :** UCI Diabetes 130-US Hospitals Dataset
([lien Kaggle](https://www.kaggle.com/datasets/jimschacko/130-us-hospitals-dataset))

| Propriété | Valeur |
|---|---|
| Observations brutes | 101 766 séjours |
| Après déduplication patient | 67 580 observations |
| Variables | 50 (démographiques, cliniques, biologiques, pharmaceutiques) |
| Période | 1999 — 2008 |
| Hôpitaux | 130 aux États-Unis |
| Variable cible | Réadmission dans les 30 jours (`<30` = 1, reste = 0) |
| Déséquilibre des classes | 9,1 % positifs / 90,9 % négatifs (ratio 8,85:1) |

### Décisions de préprocessing motivées par l'EDA

| Problème identifié | Décision |
|---|---|
| `weight` : 97 % NaN | Suppression de la colonne |
| `medical_specialty` : 49,9 % NaN | Suppression de la colonne |
| `payer_code` : 40,3 % NaN | Suppression de la colonne |
| `max_glu_serum` : 94,7 % NaN | Conversion en flags binaires `glu_tested` / `glu_high` |
| `A1Cresult` : 83,4 % NaN | Conversion en flags binaires `A1C_tested` / `A1C_high` |
| `diag_2` / `diag_3` avec `?` | NaN natifs + indicateurs `diag2_missing`, `diag3_missing` |
| Patients multi-visites | Déduplication : 1ère visite par patient uniquement |
| 19 paires corrélées `\|r\|` > 0,70 | Suppression de la feature la moins corrélée à la cible |

### Features dérivées créées (feature engineering)

```python
care_intensity    = lab_procedures + procedures + medications
med_per_day       = medications / time_in_hospital
service_utilization = outpatient + inpatient + emergency
age_x_inpatient   = age × number_inpatient        # meilleure corrélation cible
age_x_meds        = age × num_medications
polypharmacy      = (medications >= 15)            # flag binaire
long_stay         = (time_in_hospital > 7)         # flag binaire
uncontrolled_dm   = A1C_high AND diabetesMed AND change
insulin_change    = insulin AND change
n_serious_diag    = somme diagnostics Circulatoire / Respiratoire / Diabète
```

---

## Résultats du modèle

**Modèle retenu :** LightGBM avec gestion native des NaN,
optimisé sur l'AUC-PR (`average_precision`)

> **Pourquoi AUC-PR et non AUC-ROC ?**
> Avec 9,1 % de positifs, la baseline d'un classifieur aléatoire
> est AUC-ROC = 0,50 mais AUC-PR = 0,091 (= prévalence).
> L'AUC-PR mesure réellement la performance sur la classe minoritaire.

### Métriques finales sur le jeu de test

| Métrique | Valeur | Interprétation |
|---|---|---|
| **AUC-PR** | **0,1727** | ×1,9 vs baseline aléatoire (0,091) |
| AUC-ROC | 0,6535 | Cohérent avec la littérature (~0,65–0,70) |
| Recall | 62,2 % | 762 / 1 226 patients réadmis détectés |
| Precision | 12,5 % | 1 alarme correcte sur 8 |
| F1-Score | 0,208 | — |
| Seuil médical | 0,098 | Optimisé pour Recall ≥ 60 % |
| Gap CV-Test | < 0,01 | Aucun surapprentissage |

### Comparaison des modèles

| Modèle | AUC-ROC | AUC-PR | Gap CV-Test |
|---|---|---|---|
| Régression Logistique | 0,635 | 0,134 | 0,000 |
| XGBoost | 0,641 | 0,149 | 0,000 |
| LightGBM | 0,641 | 0,157 | 0,000 |
| XGBoost (NaN natif) | 0,658 | 0,168 | < 0,01 |
| **LightGBM (NaN natif)** | **0,654** | **0,1727** | **< 0,01** |

> **Impact de l'approche NaN natif :**
> Conserver les NaN plutôt que supprimer les lignes apporte
> **+11 % d'AUC-PR** (0,1727 vs 0,1554) en exploitant
> le signal clinique de l'absence d'information.

### Top 5 features SHAP (importance globale)

| Rang | Feature | Description |
|---|---|---|
| 1 | `discharge_disposition_id` | Destination de sortie |
| 2 | `age_x_inpatient` | Âge × hospitalisations antérieures |
| 3 | `age` | Tranche d'âge du patient |
| 4 | `numchange` | Nb de médicaments modifiés |
| 5 | `time_in_hospital` | Durée du séjour |

---

## Architecture du projet
# Prédiction de Réadmission Hospitalière — Pipeline MLOps

> Système d'aide à la décision médicale prédit le risque de réadmission
> hospitalière dans les 30 jours pour patients diabétiques.
> Pipeline MLOps complet : DVC · MLflow · FastAPI · Streamlit · Docker · CI/CD

---

## Table des matières

- [Contexte](#contexte)
- [Dataset](#dataset)
- [Résultats du modèle](#résultats-du-modèle)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Lancer le pipeline DVC](#lancer-le-pipeline-dvc)
- [MLflow — Tracking des expériences](#mlflow--tracking-des-expériences)
- [API REST](#api-rest)
- [Interface Streamlit](#interface-streamlit)
- [Tests automatiques](#tests-automatiques)
- [Docker](#docker)
- [Structure du dépôt](#structure-du-dépôt)

---

## Contexte

La réadmission hospitalière non planifiée dans les 30 jours suivant
une sortie représente un indicateur clé de la qualité des soins.
Aux États-Unis, les réadmissions évitables génèrent **26 milliards
de dollars de dépenses annuelles**.

Ce projet construit un pipeline MLOps complet qui :

- Prédit le risque de réadmission à partir de données cliniques tabulaires
- Explique chaque prédiction via SHAP (facteurs de risque individuels)
- Expose le modèle via une API REST documentée
- Fournit une interface médicale Streamlit accessible sans compétence technique
- Garantit la reproductibilité via DVC et la traçabilité via MLflow
- Automatise les tests et le déploiement via GitHub Actions

---

## Dataset

**Source :** UCI Diabetes 130-US Hospitals Dataset
([lien Kaggle](https://www.kaggle.com/datasets/jimschacko/130-us-hospitals-dataset))

| Propriété | Valeur |
|---|---|
| Observations brutes | 101 766 séjours |
| Après déduplication patient | 67 580 observations |
| Variables | 50 (démographiques, cliniques, biologiques, pharmaceutiques) |
| Période | 1999 — 2008 |
| Hôpitaux | 130 aux États-Unis |
| Variable cible | Réadmission dans les 30 jours (`<30` = 1, reste = 0) |
| Déséquilibre des classes | 9,1 % positifs / 90,9 % négatifs (ratio 8,85:1) |

### Décisions de préprocessing motivées par l'EDA

| Problème identifié | Décision |
|---|---|
| `weight` : 97 % NaN | Suppression de la colonne |
| `medical_specialty` : 49,9 % NaN | Suppression de la colonne |
| `payer_code` : 40,3 % NaN | Suppression de la colonne |
| `max_glu_serum` : 94,7 % NaN | Conversion en flags binaires `glu_tested` / `glu_high` |
| `A1Cresult` : 83,4 % NaN | Conversion en flags binaires `A1C_tested` / `A1C_high` |
| `diag_2` / `diag_3` avec `?` | NaN natifs + indicateurs `diag2_missing`, `diag3_missing` |
| Patients multi-visites | Déduplication : 1ère visite par patient uniquement |
| 19 paires corrélées `\|r\|` > 0,70 | Suppression de la feature la moins corrélée à la cible |

### Features dérivées créées (feature engineering)

```python
care_intensity    = lab_procedures + procedures + medications
med_per_day       = medications / time_in_hospital
service_utilization = outpatient + inpatient + emergency
age_x_inpatient   = age × number_inpatient        # meilleure corrélation cible
age_x_meds        = age × num_medications
polypharmacy      = (medications >= 15)            # flag binaire
long_stay         = (time_in_hospital > 7)         # flag binaire
uncontrolled_dm   = A1C_high AND diabetesMed AND change
insulin_change    = insulin AND change
n_serious_diag    = somme diagnostics Circulatoire / Respiratoire / Diabète
```

---

## Résultats du modèle

**Modèle retenu :** LightGBM avec gestion native des NaN,
optimisé sur l'AUC-PR (`average_precision`)

> **Pourquoi AUC-PR et non AUC-ROC ?**
> Avec 9,1 % de positifs, la baseline d'un classifieur aléatoire
> est AUC-ROC = 0,50 mais AUC-PR = 0,091 (= prévalence).
> L'AUC-PR mesure réellement la performance sur la classe minoritaire.

### Métriques finales sur le jeu de test

| Métrique | Valeur | Interprétation |
|---|---|---|
| **AUC-PR** | **0,1727** | ×1,9 vs baseline aléatoire (0,091) |
| AUC-ROC | 0,6535 | Cohérent avec la littérature (~0,65–0,70) |
| Recall | 62,2 % | 762 / 1 226 patients réadmis détectés |
| Precision | 12,5 % | 1 alarme correcte sur 8 |
| F1-Score | 0,208 | — |
| Seuil médical | 0,098 | Optimisé pour Recall ≥ 60 % |
| Gap CV-Test | < 0,01 | Aucun surapprentissage |

### Comparaison des modèles

| Modèle | AUC-ROC | AUC-PR | Gap CV-Test |
|---|---|---|---|
| Régression Logistique | 0,635 | 0,134 | 0,000 |
| XGBoost | 0,641 | 0,149 | 0,000 |
| LightGBM | 0,641 | 0,157 | 0,000 |
| XGBoost (NaN natif) | 0,658 | 0,168 | < 0,01 |
| **LightGBM (NaN natif)** | **0,654** | **0,1727** | **< 0,01** |

> **Impact de l'approche NaN natif :**
> Conserver les NaN plutôt que supprimer les lignes apporte
> **+11 % d'AUC-PR** (0,1727 vs 0,1554) en exploitant
> le signal clinique de l'absence d'information.

### Top 5 features SHAP (importance globale)

| Rang | Feature | Description |
|---|---|---|
| 1 | `discharge_disposition_id` | Destination de sortie |
| 2 | `age_x_inpatient` | Âge × hospitalisations antérieures |
| 3 | `age` | Tranche d'âge du patient |
| 4 | `numchange` | Nb de médicaments modifiés |
| 5 | `time_in_hospital` | Durée du séjour |

---
---

## Installation

### Prérequis

- Python 3.10+
- Git
- Docker Desktop (pour le déploiement conteneurisé)
- DVC (`pip install dvc`)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/ayoub-mourou/pfa_readmission_ml.git
cd pfa_readmission_ml

# 2. Créer et activer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Copier le dataset brut
cp /chemin/vers/diabetic_data.csv data/raw/diabetic_data.csv
```

---

## Lancer le pipeline DVC

```bash
# Visualiser le DAG du pipeline
dvc dag

# Lancer le pipeline complet (preprocess → train → evaluate)
dvc repro

# Afficher les métriques après exécution
dvc metrics show

# Comparer avec le run précédent
dvc metrics diff

# Tracker les données avec DVC
dvc add data/raw/diabetic_data.csv
dvc push
```

Le pipeline DVC comporte **3 stages** définis dans `dvc.yaml` :

| Stage | Entrées | Sorties |
|---|---|---|
| `preprocess` | `diabetic_data.csv` · `params.yaml` | `X_train.pkl` · `X_test.pkl` · `y_train.pkl` · `y_test.pkl` |
| `train` | `X_train.pkl` · `y_train.pkl` | `best_model.pkl` · `feature_names.pkl` |
| `evaluate` | `X_test.pkl` · `y_test.pkl` · `best_model.pkl` | `metrics.json` · `confusion_matrix.png` · `threshold.pkl` |

> Modifier un paramètre dans `params.yaml` et relancer `dvc repro`
> ne ré-exécutera que les stages affectés par ce changement.

---

## MLflow — Tracking des expériences

```bash
# Lancer l'interface MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Ouvrir [http://localhost:5000](http://localhost:5000)

Chaque run enregistre automatiquement :

- **Paramètres** : tous les hyperparamètres LightGBM retenus
- **Métriques** : AUC-PR, AUC-ROC, Recall, Precision, F1, TP, FP, FN, TN
- **Artefacts** : modèle sérialisé, matrice de confusion, `metrics.json`

---

## API REST

### Démarrer l'API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API et disponibilité du modèle |
| `POST` | `/predict` | Prédiction + probabilité + SHAP top 3 |
| `GET` | `/model-info` | Métriques, version et métadonnées |
| `GET` | `/metrics` | Métriques Prometheus |
| `GET` | `/docs` | Documentation Swagger interactive |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_inpatient": 1,
    "number_emergency": 0,
    "number_diagnoses": 7,
    "age": "[70-80)",
    "gender": "Male",
    "race": "Caucasian",
    "diag_1": "410",
    "diag_2": "250",
    "diag_3": "401",
    "max_glu_serum": "None",
    "A1Cresult": ">7",
    "change": "Ch",
    "diabetesMed": "Yes",
    "insulin": "Up",
    "metformin": "Steady"
  }'
```

### Exemple de réponse

```json
{
  "prediction": 1,
  "probability": 0.1342,
  "risk_level": "Élevé",
  "threshold": 0.0985,
  "label": "Réadmission probable (<30j)",
  "top_risk_factors": [
    {
      "feature": "age_x_inpatient",
      "shap_value": 0.61,
      "direction": "↑ risque"
    },
    {
      "feature": "discharge_disposition_id",
      "shap_value": 0.31,
      "direction": "↑ risque"
    },
    {
      "feature": "numchange",
      "shap_value": -0.18,
      "direction": "↓ risque"
    }
  ]
}
```

### Validation des entrées (Pydantic)

L'API valide chaque champ avant de contacter le modèle.
En cas de données invalides, une erreur `422` est retournée :

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "age invalide '[999-1000)'. Attendu : ['[0-10)', '[10-20)', ...]",
      "type": "value_error"
    }
  ]
}
```

---

## Interface Streamlit

```bash
# Installer les dépendances frontend
pip install -r frontend/requirements_fe.txt

# Lancer l'interface (l'API doit être active)
streamlit run frontend/app_ui.py --server.port 8501
```

Ouvrir [http://localhost:8501](http://localhost:8501)

L'interface propose :

- **Formulaire patient** organisé en colonnes thématiques
  (démographie, données cliniques, historique, traitement)
- **Score de risque** avec code couleur
  (rouge = Élevé, orange = Modéré, vert = Faible)
- **Graphique SHAP local** : les 3 facteurs ayant le plus
  influencé la décision pour ce patient
- **Panneau d'interprétation clinique** avec métriques du modèle

---

## Tests automatiques

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ -v --cov=src --cov-report=term-missing

# Fichier spécifique
pytest tests/test_api.py -v
```

### Couverture des tests

| Fichier | Tests | Ce qui est vérifié |
|---|---|---|
| `test_data_loader.py` | 6 | Encodage, feature engineering, flags NaN |
| `test_preprocessing.py` | 5 | Shapes, stratification, absence de NaN |
| `test_api.py` | 8 | Endpoints, validation Pydantic, SHAP |
| **Total** | **19** | **100 % passent** |

---

## Docker

### Développement (API + Frontend)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Production (images Docker Hub)

```bash
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Build manuel

```bash
# Image API
docker build -f docker/Dockerfile.api -t readmission-api .
docker run -p 8000:8000 readmission-api

# Image Frontend
docker build -f docker/Dockerfile.frontend -t readmission-frontend .
docker run -p 8501:8501 readmission-frontend
```

### Services Docker

| Conteneur | Image | Port | Rôle |
|---|---|---|---|
| `readmission_api` | `python:3.10-slim` | 8000 | FastAPI + LightGBM |
| `readmission_frontend` | `python:3.10-slim` | 8501 | Streamlit |

---
---

## Accès rapide

| Service | URL | Commande |
|---|---|---|
| API Swagger | http://localhost:8000/docs | `uvicorn api.app:app --reload` |
| Interface médicale | http://localhost:8501 | `streamlit run frontend/app_ui.py` |
| MLflow UI | http://localhost:5000 | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

---
