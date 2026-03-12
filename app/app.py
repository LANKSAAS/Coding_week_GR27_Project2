"""
app.py — Streamlit Web Application for Obesity Risk Estimation.

A professional clinical-decision-support wizard that guides physicians
through three clinical steps, collects patient data, then delivers an
obesity-level prediction with SHAP explanations.

Run with:
    streamlit run app/app.py
"""

import os
import sys
import base64
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Ensure imports from src/ work regardless of cwd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data_processing import fetch_dataset, preprocess_data

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# ---------------------------------------------------------------------------
# Helpers — image loading
# ---------------------------------------------------------------------------
def _load_img_b64(filename: str) -> str:
    """Return base64-encoded <img> tag if the file exists, else ''."""
    path = os.path.join(ASSETS, filename)
    if os.path.exists(path):
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'<img src="data:image/png;base64,{data}" />'
    return ""


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Obesity Risk Estimation — GR 27",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state — wizard step management
# ---------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0  # 0=welcome, 1/2/3=steps, 4=results

# Persist all inputs in session state with defaults
_DEFAULTS = {
    "w_gender": "Male", "w_age": 25, "w_height": 1.70, "w_weight": 70.0,
    "w_favc": "yes", "w_fcvc": 2.0, "w_ncp": 3.0,
    "w_caec": "Sometimes", "w_ch2o": 2.0, "w_calc": "no",
    "w_scc": "no", "w_faf": 1.0, "w_tue": 1.0,
    "w_smoke": "no", "w_family": "yes",
    "w_mtrans": "Public_Transportation",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_next():
    st.session_state.step += 1

def go_back():
    st.session_state.step -= 1

def go_home():
    st.session_state.step = 0


# ---------------------------------------------------------------------------
# Custom CSS — premium medical palette with readable step labels
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a237e 0%, #1565c0 50%, #1e88e5 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(21,101,192,0.25);
    }
    .header-banner img { height: 70px; border-radius: 8px; }
    .header-banner h1 { color: #ffffff; margin: 0; font-size: 1.6rem; font-weight: 700; }
    .header-banner p  { color: #bbdefb; margin: 0; font-size: 0.95rem; }

    /* Step title */
    .step-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 1rem 0 0.3rem 0;
        letter-spacing: 0.3px;
    }
    .step-title.blue  { color: #1565c0; }
    .step-title.green { color: #2e7d32; }
    .step-title.orange { color: #e65100; }

    /* Step subtitle */
    .step-subtitle {
        text-align: center;
        font-size: 1.0rem;
        color: #546e7a;
        margin-bottom: 1.2rem;
    }

    /* Progress bar container */
    .progress-bar {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .progress-dot {
        width: 38px; height: 38px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.95rem;
        color: #fff;
        transition: all 0.3s;
    }
    .progress-dot.active   { background: #1565c0; box-shadow: 0 0 12px rgba(21,101,192,0.5); }
    .progress-dot.done     { background: #2e7d32; }
    .progress-dot.pending  { background: #b0bec5; }
    .progress-line {
        width: 50px; height: 4px;
        border-radius: 2px;
        align-self: center;
    }
    .progress-line.done    { background: #2e7d32; }
    .progress-line.pending { background: #cfd8dc; }

    /* Step image */
    .step-image { text-align: center; margin: 1rem 0; }
    .step-image img { max-height: 220px; border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.10); }

    /* Cards */
    .metric-card {
        background: #ffffff; padding: 1.4rem; border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center;
        border-left: 4px solid #1565c0;
    }
    .metric-card h2 { color: #1a237e; margin: 0 0 0.3rem 0; font-size: 1.1rem; }
    .metric-card .value { color: #1565c0; font-size: 1.8rem; font-weight: 700; }

    /* Section titles */
    .section-title { color: #1a237e; font-size: 1.2rem; font-weight: 600;
        margin: 1.2rem 0 0.6rem 0; border-bottom: 2px solid #1565c0; padding-bottom: 0.3rem; }

    /* Welcome hero */
    .welcome-hero { text-align: center; padding: 2rem 1rem; }
    .welcome-hero h2 { color: #1a237e; font-size: 1.8rem; margin-bottom: 0.5rem; }
    .welcome-hero p { color: #546e7a; font-size: 1.05rem; max-width: 700px;
        margin: 0 auto 1.5rem auto; line-height: 1.6; }
    .welcome-hero img { border-radius: 14px; max-height: 340px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10); margin-bottom: 1.5rem; }

    /* Medical report box */
    .report-box {
        background: #ffffff; border-left: 5px solid #1565c0;
        border-radius: 8px; padding: 1.2rem 1.6rem; margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #37474f; font-size: 0.95rem; line-height: 1.7;
    }
    .report-box h3 { color: #1a237e; margin: 0 0 0.5rem 0; font-size: 1.05rem; }

    /* Form container */
    .form-container {
        background: #ffffff;
        border-radius: 14px;
        padding: 2rem 2.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        max-width: 750px;
        margin: 0 auto 1.5rem auto;
    }

    /* Hide sidebar entirely */
    section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------
logo_html = _load_img_b64("ecc_logo.png")
st.markdown(f"""
<div class="header-banner">
    {logo_html}
    <div>
        <h1>🏥 Obesity Risk Estimation</h1>
        <p>Clinical Decision Support System — Ecole Centrale Casablanca - GR 27</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model artefacts
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(DATA_DIR, "best_model.joblib"))

@st.cache_resource
def load_encoders():
    return joblib.load(os.path.join(DATA_DIR, "label_encoders.joblib"))

@st.cache_resource
def load_feature_columns():
    return joblib.load(os.path.join(DATA_DIR, "feature_columns.joblib"))

@st.cache_resource
def load_background_data():
    df = fetch_dataset()
    _, _, _, _, le, fc = preprocess_data(df)
    df2 = df.copy()
    for col, enc in le.items():
        if col == "target":
            continue
        if col in df2.columns:
            df2[col] = enc.transform(df2[col])
    df2["NObeyesdad"] = le["target"].transform(df2["NObeyesdad"])
    X = df2[fc]
    return X.sample(n=min(200, len(X)), random_state=42)

try:
    model = load_model()
    label_encoders = load_encoders()
    feature_columns = load_feature_columns()
    bg_data = load_background_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model artefacts. Run `python src/train_model.py` first.\n\n{e}")

if not model_loaded:
    st.stop()

le_target = label_encoders["target"]
class_names = list(le_target.classes_)

# ---------------------------------------------------------------------------
# Helper — progress indicator
# ---------------------------------------------------------------------------
def render_progress(current_step: int):
    """Show a 3-dot progress bar (steps 1-2-3)."""
    dots = []
    for i in range(1, 4):
        if i < current_step:
            cls = "done"
        elif i == current_step:
            cls = "active"
        else:
            cls = "pending"
        dots.append(f'<div class="progress-dot {cls}">{i}</div>')
        if i < 3:
            line_cls = "done" if i < current_step else "pending"
            dots.append(f'<div class="progress-line {line_cls}"></div>')
    st.markdown(f'<div class="progress-bar">{"".join(dots)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STEP 0 — Welcome / Landing
# ---------------------------------------------------------------------------
if st.session_state.step == 0:
    doctor_html = _load_img_b64("doctor_image.png")
    st.markdown(f"""
    <div class="welcome-hero">
        {doctor_html}
        <h2>Bienvenue — Système d'Estimation du Risque d'Obésité</h2>
        <p>
            Cet outil d'aide à la décision clinique permet aux médecins d'évaluer le
            niveau de risque d'obésité d'un patient à partir de données biométriques,
            d'habitudes alimentaires et de mode de vie.<br><br>
            Répondez à <strong>3 étapes simples</strong>, puis recevez une prédiction
            alimentée par l'IA avec des explications SHAP interprétables.
        </p>
        <p style="color:#1565c0; font-weight:600; font-size:0.95rem;">
            Powered by LightGBM &middot; SHAP Explainability &middot; Ecole Centrale Casablanca
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.button("🩺  Commencer le Diagnostic", on_click=go_next,
                  use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# STEP 1 — Profil Biométrique
# ---------------------------------------------------------------------------
elif st.session_state.step == 1:
    render_progress(1)

    step1_img = _load_img_b64("step1_biometric.png")
    st.markdown(f"""
    <div class="step-title blue">📏  Étape 1 — Profil Biométrique</div>
    <div class="step-subtitle">Renseignez les données physiques du patient</div>
    <div class="step-image">{step1_img}</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.session_state.w_gender = st.selectbox(
            "👤 Genre", ["Male", "Female"],
            index=["Male", "Female"].index(st.session_state.w_gender),
        )
        st.session_state.w_age = st.slider("🎂 Âge", 5, 100, st.session_state.w_age)
        st.session_state.w_height = st.slider(
            "📐 Taille (m)", 1.20, 2.30, st.session_state.w_height, step=0.01)
        st.session_state.w_weight = st.slider(
            "⚖️ Poids (kg)", 20.0, 250.0, st.session_state.w_weight, step=0.5)

        st.markdown("")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.button("⬅️  Retour", on_click=go_back, use_container_width=True)
        with bc2:
            st.button("Suivant  ➡️", on_click=go_next, use_container_width=True,
                      type="primary")

# ---------------------------------------------------------------------------
# STEP 2 — Habitudes Alimentaires
# ---------------------------------------------------------------------------
elif st.session_state.step == 2:
    render_progress(2)

    step2_img = _load_img_b64("step2_eating.png")
    st.markdown(f"""
    <div class="step-title green">🍽️  Étape 2 — Habitudes Alimentaires</div>
    <div class="step-subtitle">Décrivez les habitudes alimentaires du patient</div>
    <div class="step-image">{step2_img}</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.session_state.w_favc = st.selectbox(
            "🍔 Nourriture calorique fréquente (FAVC) ?", ["yes", "no"],
            index=["yes", "no"].index(st.session_state.w_favc),
        )
        st.session_state.w_fcvc = st.slider(
            "🥦 Fréquence de légumes (FCVC)", 0.0, 5.0,
            st.session_state.w_fcvc, step=0.1)
        st.session_state.w_ncp = st.slider(
            "🍽️ Repas principaux / jour (NCP)", 1.0, 8.0,
            st.session_state.w_ncp, step=0.1)
        _caec_opts = ["no", "Sometimes", "Frequently", "Always"]
        st.session_state.w_caec = st.selectbox(
            "🍪 Grignotage entre repas (CAEC)", _caec_opts,
            index=_caec_opts.index(st.session_state.w_caec),
        )
        st.session_state.w_ch2o = st.slider(
            "💧 Eau quotidienne (CH2O, litres)", 0.5, 6.0,
            st.session_state.w_ch2o, step=0.1)
        _calc_opts = ["no", "Sometimes", "Frequently", "Always"]
        st.session_state.w_calc = st.selectbox(
            "🍷 Consommation d'alcool (CALC)", _calc_opts,
            index=_calc_opts.index(st.session_state.w_calc),
        )

        st.markdown("")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.button("⬅️  Retour", on_click=go_back, use_container_width=True)
        with bc2:
            st.button("Suivant  ➡️", on_click=go_next, use_container_width=True,
                      type="primary")

# ---------------------------------------------------------------------------
# STEP 3 — Mode de Vie & Antécédents
# ---------------------------------------------------------------------------
elif st.session_state.step == 3:
    render_progress(3)

    step3_img = _load_img_b64("step3_lifestyle.png")
    st.markdown(f"""
    <div class="step-title orange">🏃  Étape 3 — Mode de Vie & Antécédents</div>
    <div class="step-subtitle">Renseignez le mode de vie et les antécédents du patient</div>
    <div class="step-image">{step3_img}</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.session_state.w_scc = st.selectbox(
            "📊 Suivi des calories (SCC) ?", ["yes", "no"],
            index=["yes", "no"].index(st.session_state.w_scc),
        )
        st.session_state.w_faf = st.slider(
            "🏋️ Activité physique (FAF, jours/sem)", 0.0, 7.0,
            st.session_state.w_faf, step=0.1)
        st.session_state.w_tue = st.slider(
            "📱 Temps d'écran (TUE, heures/jour)", 0.0, 12.0,
            st.session_state.w_tue, step=0.1)
        st.session_state.w_smoke = st.selectbox(
            "🚬 Fumeur (SMOKE) ?", ["yes", "no"],
            index=["yes", "no"].index(st.session_state.w_smoke),
        )
        st.session_state.w_family = st.selectbox(
            "👨‍👩‍👧 Antécédents familiaux de surpoids ?", ["yes", "no"],
            index=["yes", "no"].index(st.session_state.w_family),
        )
        _mtrans_opts = ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]
        st.session_state.w_mtrans = st.selectbox(
            "🚗 Transport principal (MTRANS)", _mtrans_opts,
            index=_mtrans_opts.index(st.session_state.w_mtrans),
        )

        st.markdown("")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.button("⬅️  Retour", on_click=go_back, use_container_width=True)
        with bc2:
            st.button("🔍  Predict Obesity Risk", on_click=go_next,
                      use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# STEP 4 — Results
# ---------------------------------------------------------------------------
elif st.session_state.step == 4:
    # ---- Encode inputs from session state ----
    gender = st.session_state.w_gender
    age = st.session_state.w_age
    height = st.session_state.w_height
    weight = st.session_state.w_weight
    favc = st.session_state.w_favc
    fcvc = st.session_state.w_fcvc
    ncp = st.session_state.w_ncp
    caec = st.session_state.w_caec
    ch2o = st.session_state.w_ch2o
    calc = st.session_state.w_calc
    scc = st.session_state.w_scc
    faf = st.session_state.w_faf
    tue = st.session_state.w_tue
    smoke = st.session_state.w_smoke
    family = st.session_state.w_family
    mtrans = st.session_state.w_mtrans

    raw = {
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history_with_overweight": family, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "SMOKE": smoke,
        "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
        "CALC": calc, "MTRANS": mtrans,
    }
    row = pd.DataFrame([raw])
    for col in row.columns:
        if col in label_encoders and col != "target":
            le = label_encoders[col]
            row[col] = le.transform(row[col].astype(str))
    input_df = row[feature_columns].apply(pd.to_numeric)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]
    confidence = probabilities.max() * 100
    bmi_val = weight / (height ** 2)

    # ---- TOP: Result cards ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>Predicted Level</h2>
            <div class="value">{predicted_label}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>Confidence</h2>
            <div class="value">{confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>BMI</h2>
            <div class="value">{bmi_val:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ---- BMI category ----
    if bmi_val < 18.5:
        bmi_category = "Underweight"
    elif bmi_val < 25:
        bmi_category = "Normal weight"
    elif bmi_val < 30:
        bmi_category = "Overweight"
    elif bmi_val < 35:
        bmi_category = "Obesity Class I"
    elif bmi_val < 40:
        bmi_category = "Obesity Class II"
    else:
        bmi_category = "Obesity Class III (Severe)"

    # ---- Clinical summary ----
    st.markdown(f"""
    <div class="report-box">
        <h3>📝 Résumé Clinique</h3>
        <strong>Profil patient :</strong> {gender}, {age} ans, {height:.2f} m, {weight:.1f} kg<br>
        <strong>BMI calculé :</strong> {bmi_val:.1f} ({bmi_category})<br>
        <strong>Prédiction ML :</strong> <em>{predicted_label}</em> (confiance : {confidence:.1f}%)<br><br>
        <strong>Recommandation :</strong>
        Cette prédiction est générée par un modèle de machine learning entraîné sur des
        données de style de vie et de condition physique. Elle doit être utilisée comme
        <em>aide au dépistage</em> en complément du jugement clinique.
        Les facteurs clés influençant cette prédiction sont présentés dans l'explication SHAP ci-dessous.
    </div>
    """, unsafe_allow_html=True)

    # ---- Charts row ----
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">📊 Probabilités de Prédiction</div>',
                    unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Obesity Level": class_names,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=True)

        colors = ["#1565c0" if lbl == predicted_label else "#90caf9"
                  for lbl in prob_df["Obesity Level"]]

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Obesity Level"],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in prob_df["Probability"]],
            textposition="auto",
            textfont=dict(color="#1a237e", size=13),
        ))
        fig.update_layout(
            xaxis_title="Probability",
            yaxis_title="",
            template="plotly_white",
            height=400,
            margin=dict(l=10, r=50, t=20, b=40),
            xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            yaxis=dict(tickfont=dict(size=12)),
            font=dict(color="#1a237e"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">🔬 Explication SHAP</div>',
                    unsafe_allow_html=True)
        try:
            explainer = shap.TreeExplainer(model, bg_data)
            shap_values = explainer(input_df)
            if shap_values.values.ndim == 3:
                sv = shap_values[0, :, prediction]
            else:
                sv = shap_values[0]
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(sv, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except Exception as exc:
            st.warning(f"SHAP visualisation unavailable: {exc}")

    # ---- Restart button ----
    st.markdown("---")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.button("🏠  Nouveau Diagnostic", on_click=go_home,
                  use_container_width=True, type="primary")
