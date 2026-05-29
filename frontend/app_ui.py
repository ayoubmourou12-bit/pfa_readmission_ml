import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

API_URL = "http://localhost:8000"

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Réadmission Hospitalière — Aide à la Décision",
    page_icon="🏥",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Prédiction de Réadmission Hospitalière")
st.caption(
    "Outil d'aide à la décision médicale — "
    "Prédit le risque de réadmission dans les 30 jours · "
    "Modèle : LightGBM (AUC-PR optimisé)"
)

# ── Statut API ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Statut du système")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API en ligne ✅  |  Modèle chargé : {health['model_loaded']}")
    except Exception:
        st.error("API hors ligne ❌ — Lancer `uvicorn api.app:app`")

    st.divider()
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        st.metric("AUC-PR",  info["metrics"].get("auc_pr", "N/A"))
        st.metric("AUC-ROC", info["metrics"].get("auc_roc", "N/A"))
        st.metric("Recall",  info["metrics"].get("recall", "N/A"))
        st.metric("Features", info["n_features"])
        st.caption(
            f"Seuil : {info['metrics'].get('threshold','N/A')} · "
            f"Prevalence : {info['metrics'].get('baseline_auc_pr','N/A')}"
        )
    except Exception:
        st.warning("Métriques non disponibles")

# ── Formulaire patient ────────────────────────────────────────────────────────
st.subheader("📋 Informations du patient")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Données démographiques**")
    age    = st.selectbox("Tranche d'âge", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ], index=7)
    gender = st.radio("Genre", ["Male", "Female"], horizontal=True)
    race   = st.selectbox("Race", [
        "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
    ])

with col2:
    st.markdown("**Données cliniques du séjour**")
    time_in_hospital    = st.slider("Durée du séjour (jours)", 1, 14, 5)
    num_lab_procedures  = st.slider("Nb procédures lab",        1, 132, 45)
    num_procedures      = st.slider("Nb procédures",            0, 6, 1)
    num_medications     = st.slider("Nb médicaments",           1, 79, 18)
    number_diagnoses    = st.slider("Nb diagnostics",           1, 16, 7)

with col3:
    st.markdown("**Historique hospitalier**")
    number_outpatient = st.number_input("Visites ambulatoires antérieures", 0, 40, 0)
    number_inpatient  = st.number_input("Hospitalisations antérieures",     0, 20, 1)
    number_emergency  = st.number_input("Urgences antérieures",             0, 30, 0)

st.divider()
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**Diagnostics ICD-9**")
    diag_1 = st.text_input("Diagnostic principal", value="410")
    diag_2 = st.text_input("Diagnostic secondaire 1", value="250")
    diag_3 = st.text_input("Diagnostic secondaire 2", value="401")

with col5:
    st.markdown("**Tests biologiques**")
    max_glu_serum = st.selectbox("Glucose sérum", ["None", "Norm", ">200", ">300"])
    A1Cresult     = st.selectbox("Résultat A1C",  ["None", "Norm", ">7", ">8"])

with col6:
    st.markdown("**Traitement diabétique**")
    change      = st.radio("Changement de médication", ["No", "Ch"], horizontal=True)
    diabetesMed = st.radio("Sous médication diabète",  ["No", "Yes"], horizontal=True)
    insulin     = st.selectbox("Insuline",   ["No", "Steady", "Up", "Down"])
    metformin   = st.selectbox("Metformine", ["No", "Steady", "Up", "Down"])

# ── Bouton prédiction ─────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Prédire le risque de réadmission", type="primary",
                        use_container_width=True)

if predict_btn:
    payload = {
        "time_in_hospital":         time_in_hospital,
        "num_lab_procedures":       num_lab_procedures,
        "num_procedures":           num_procedures,
        "num_medications":          num_medications,
        "number_outpatient":        int(number_outpatient),
        "number_inpatient":         int(number_inpatient),
        "number_emergency":         int(number_emergency),
        "number_diagnoses":         number_diagnoses,
        "age":                      age,
        "gender":                   gender,
        "race":                     race,
        "diag_1":                   diag_1,
        "diag_2":                   diag_2,
        "diag_3":                   diag_3,
        "max_glu_serum":            max_glu_serum,
        "A1Cresult":                A1Cresult,
        "change":                   change,
        "diabetesMed":              diabetesMed,
        "insulin":                  insulin,
        "metformin":                metformin,
        "admission_type_id":        1,
        "discharge_disposition_id": 1,
        "admission_source_id":      7,
    }

    with st.spinner("Analyse en cours..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if resp.status_code == 200:
                result = resp.json()

                # ── Résultat principal ────────────────────────────────────────
                st.subheader("📊 Résultat de la prédiction")
                res_col1, res_col2, res_col3 = st.columns(3)

                risk_color = {
                    "Élevé":  "🔴",
                    "Modéré": "🟠",
                    "Faible": "🟢",
                }
                emoji = risk_color.get(result["risk_level"], "⚪")

                with res_col1:
                    st.metric("Risque", f"{emoji} {result['risk_level']}")
                with res_col2:
                    st.metric("Probabilité",
                              f"{result['probability']*100:.1f}%",
                              delta=f"Seuil : {result['threshold']*100:.1f}%")
                with res_col3:
                    label_color = "red" if result["prediction"] == 1 else "green"
                    st.markdown(
                        f"<h4 style='color:{label_color}'>"
                        f"{'⚠️ ' if result['prediction']==1 else '✅ '}"
                        f"{result['label']}</h4>",
                        unsafe_allow_html=True
                    )

                # ── SHAP local (top 3 facteurs de risque) ────────────────────
                top3 = result.get("top_risk_factors", [])
                if top3:
                    st.subheader("🔍 Facteurs de risque principaux (SHAP)")
                    st.caption(
                        "Ces 3 variables ont le plus influencé la prédiction "
                        "pour **ce patient**."
                    )

                    fig, ax = plt.subplots(figsize=(8, 3))
                    features_shap = [f["feature"] for f in top3]
                    values_shap   = [f["shap_value"] for f in top3]
                    colors_shap   = [
                        "#d73027" if v > 0 else "#4575b4" for v in values_shap
                    ]
                    bars = ax.barh(features_shap[::-1], values_shap[::-1],
                                   color=colors_shap[::-1], edgecolor="white",
                                   height=0.5)
                    ax.axvline(0, color="black", lw=0.8)
                    ax.set_xlabel("Valeur SHAP (impact sur le risque)")
                    ax.set_title(
                        "↑ Rouge = augmente le risque  |  ↓ Bleu = diminue le risque",
                        fontsize=9
                    )
                    ax.spines[["top","right"]].set_visible(False)
                    for bar, val in zip(bars, values_shap[::-1]):
                        ax.text(
                            bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                            f"{val:+.4f}", va="center", fontsize=9
                        )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    for f in top3:
                        dir_icon = "🔺" if f["shap_value"] > 0 else "🔻"
                        st.write(
                            f"{dir_icon} **{f['feature']}** — "
                            f"impact : `{f['shap_value']:+.4f}` ({f['direction']})"
                        )

                # ── Interprétation clinique ───────────────────────────────────
                with st.expander("💡 Interprétation clinique", expanded=False):
                    st.markdown(f"""
**Décision du modèle :**
- Probabilité estimée de réadmission dans les 30 jours : **{result['probability']*100:.1f}%**
- Seuil de décision utilisé : **{result['threshold']*100:.1f}%**
  (optimisé pour Recall ≥ 60%)

**Attention :**
> Ce système est un outil d'**aide à la décision**, pas un substitut au
> jugement clinique. La décision finale appartient au médecin.

**Performance globale du modèle :**
- AUC-PR : **0.1727** (x1.9 vs aléatoire)
- Recall à ce seuil : **62.2%** — détecte 6 patients réadmis sur 10
- Precision : **12.5%** — 1 alarme sur 8 est correcte
                    """)

            else:
                st.error(f"Erreur API ({resp.status_code}) : {resp.json()}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de contacter l'API. "
                     "Vérifiez que le serveur tourne : "
                     "`uvicorn api.app:app --port 8000`")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")