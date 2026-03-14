"""
=========================================================
AI Obesity Risk Prediction Dashboard — Multi-Step Wizard
=========================================================
"""

import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.data_processing import fetch_dataset, preprocess_data


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Obesity Risk Dashboard — GR 27",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# PREMIUM STYLES — LIGHT CREATIVE THEME
# --------------------------------------------------

st.markdown(r"""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;600;700;800&display=swap');

/* ============================================================
   CSS VARIABLES — LIGHT THEME (default)
   ============================================================ */
:root {
    --bg-gradient: linear-gradient(160deg, #f0f4ff 0%, #e6faf2 40%, #fff8ee 70%, #f5f0ff 100%);
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-heading: #0f172a;
    --text-label: #334155;
    --text-muted: #94a3b8;

    --card-bg: rgba(255,255,255,0.75);
    --card-border: rgba(255,255,255,0.7);
    --card-shadow: 0 8px 40px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);

    --header-bg: rgba(255,255,255,0.7);
    --header-border: rgba(255,255,255,0.6);
    --header-shadow: 0 2px 16px rgba(0,0,0,0.04);

    --home-card-bg: rgba(255,255,255,0.8);
    --home-card-border: rgba(255,255,255,0.7);
    --home-card-shadow: 0 12px 48px rgba(0,0,0,0.06);

    --chip-bg: linear-gradient(135deg, #f0f9ff, #ede9fe);
    --chip-color: #475569;
    --chip-border: rgba(14, 165, 233, 0.1);

    --progress-border: #e2e8f0;
    --progress-bg: #fff;
    --progress-color: #94a3b8;
    --progress-line-bg: #e2e8f0;

    --img-shadow: 0 4px 20px rgba(0,0,0,0.08);

    --footer-color: #94a3b8;
}


/* ============================================================
   CSS VARIABLES — DARK THEME (applied via JS class)
   ============================================================ */
body.dark-theme {
    --bg-gradient: linear-gradient(160deg, #0f172a 0%, #1a1040 40%, #0c1e3a 70%, #151030 100%);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-heading: #f1f5f9;
    --text-label: #cbd5e1;
    --text-muted: #64748b;

    --card-bg: rgba(30, 41, 59, 0.7);
    --card-border: rgba(51, 65, 85, 0.5);
    --card-shadow: 0 8px 40px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.15);

    --header-bg: rgba(30, 41, 59, 0.75);
    --header-border: rgba(51, 65, 85, 0.4);
    --header-shadow: 0 2px 16px rgba(0,0,0,0.2);

    --home-card-bg: rgba(30, 41, 59, 0.8);
    --home-card-border: rgba(51, 65, 85, 0.5);
    --home-card-shadow: 0 12px 48px rgba(0,0,0,0.3);

    --chip-bg: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(139,92,246,0.12));
    --chip-color: #cbd5e1;
    --chip-border: rgba(139, 92, 246, 0.2);

    --progress-border: #334155;
    --progress-bg: #1e293b;
    --progress-color: #64748b;
    --progress-line-bg: #334155;

    --img-shadow: 0 4px 20px rgba(0,0,0,0.25);

    --footer-color: #64748b;
}

/* ---------- General ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-gradient) !important;
    font-family: 'Inter', sans-serif;
}
/* Apply text color only to Streamlit content blocks, not body */
[data-testid="stAppViewContainer"] .block-container,
[data-testid="stAppViewContainer"] .stMarkdown {
    color: var(--text-primary);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ---------- Persistent Header ---------- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 0.8rem 1rem;
    margin: -1rem auto 0.5rem auto;
    max-width: 700px;
    background: var(--header-bg);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    box-shadow: var(--header-shadow);
    border: 1px solid var(--header-border);
}
.app-header img {
    height: 42px;
    border-radius: 8px;
}
.app-header .header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.3px;
}
.app-header .header-group {
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
}

/* ---------- Progress Steps ---------- */
.progress-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin: 0.8rem auto 1.5rem auto;
    max-width: 380px;
}
.progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    flex: 1;
}
.progress-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    z-index: 2;
    transition: all 0.3s;
    border: 3px solid var(--progress-border);
    background: var(--progress-bg);
    color: var(--progress-color);
}
.progress-circle.active {
    background: linear-gradient(135deg, #38bdf8, #a78bfa);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
}
.progress-circle.done {
    background: linear-gradient(135deg, #34d399, #38bdf8);
    color: #fff;
    border-color: transparent;
}
.progress-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--progress-color);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.progress-label.active { color: #38bdf8; }
.progress-label.done { color: #34d399; }
.progress-line {
    position: absolute;
    top: 18px;
    left: 50%;
    width: 100%;
    height: 3px;
    background: var(--progress-line-bg);
    z-index: 1;
}
.progress-line.done { background: linear-gradient(90deg, #34d399, #38bdf8); }

/* ---------- Step Card ---------- */
.step-card {
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 2rem 2rem 1.5rem 2rem;
    box-shadow: var(--card-shadow);
    border: 1px solid var(--card-border);
    margin: 0 auto 1rem auto;
    max-width: 700px;
}

/* ---------- Step Header ---------- */
.step-header {
    text-align: center;
    margin-bottom: 1rem;
}
.step-emoji {
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}
.step-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--text-heading);
    margin: 0;
    letter-spacing: -0.3px;
}
.step-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 2px;
}

/* ---------- Step Image ---------- */
.step-image-container {
    display: flex;
    justify-content: center;
    margin-bottom: 1.2rem;
}
.step-image-container img {
    max-height: 140px;
    border-radius: 14px;
    box-shadow: var(--img-shadow);
    object-fit: cover;
}

/* ---------- Home Page ---------- */
.home-card {
    background: var(--home-card-bg);
    backdrop-filter: blur(16px);
    border-radius: 28px;
    padding: 2.5rem 2rem;
    box-shadow: var(--home-card-shadow);
    border: 1px solid var(--home-card-border);
    margin: 1.5rem auto;
    max-width: 600px;
    text-align: center;
}
.home-logo {
    margin-bottom: 1rem;
}
.home-logo img {
    height: 100px;
    border-radius: 14px;
    box-shadow: var(--img-shadow);
}
.home-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.home-badge {
    display: inline-block;
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    padding: 6px 24px;
    border-radius: 30px;
    letter-spacing: 2px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
}
.home-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 440px;
    margin: 0 auto 1.5rem auto;
}
.home-features {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.feature-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--chip-bg);
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--chip-color);
    border: 1px solid var(--chip-border);
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 2.2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 18px rgba(56, 189, 248, 0.25) !important;
    transition: all 0.25s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4) !important;
}
div.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ---------- Input refinements ---------- */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-weight: 500 !important;
    color: var(--text-label) !important;
    font-size: 0.9rem !important;
}

/* ---------- Result Section Titles ---------- */
.result-section {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-heading);
    margin-top: 1.8rem;
    margin-bottom: 0.4rem;
}

/* ---------- Divider ---------- */
.fancy-divider {
    height: 3px;
    background: linear-gradient(90deg, transparent, #38bdf8, #a78bfa, transparent);
    border: none;
    border-radius: 2px;
    margin: 1.5rem 0;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    color: var(--footer-color);
    font-size: 0.8rem;
    padding: 1rem 0 2rem 0;
    line-height: 1.6;
}

/* ---------- Fade-in animation ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease-out;
}
</style>

<!-- Hidden theme probe: reset all styles so it inherits ONLY Streamlit native text color -->
<div id="theme-probe" style="all:initial;position:fixed;top:-9999px;left:-9999px;pointer-events:none;visibility:hidden;color:inherit !important;">probe</div>

<script>
// Detect Streamlit dark/light theme using a hidden probe element
(function() {
    function detectTheme() {
        var probe = document.getElementById('theme-probe');
        if (!probe) return;
        // The probe inherits Streamlit's default text color
        var color = getComputedStyle(probe).color;
        var match = color.match(/\d+/g);
        if (match && match.length >= 3) {
            var r = parseInt(match[0]), g = parseInt(match[1]), b = parseInt(match[2]);
            var luminance = (0.299 * r + 0.587 * g + 0.114 * b);
            // In dark theme, text color is light (high luminance)
            // In light theme, text color is dark (low luminance)
            if (luminance > 128) {
                document.body.classList.add('dark-theme');
            } else {
                document.body.classList.remove('dark-theme');
            }
        }
    }

    setTimeout(detectTheme, 200);
    setInterval(detectTheme, 300);

    var observer = new MutationObserver(function() { setTimeout(detectTheme, 50); });
    observer.observe(document.documentElement, {
        attributes: true, subtree: true, childList: true, attributeFilter: ['style', 'class']
    });
})();
</script>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE INITIALISATION
# --------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step = 0

defaults = {
    "age": 25, "height": 1.70, "weight": 70, "gender": "Male",
    "family_history": "yes", "favc": "yes", "fcvc": 2.0,
    "ncp": 3.0, "ch2o": 2.0, "caec": "Sometimes", "scc": "no",
    "faf": 1.0, "tue": 1.0, "smoke": "no",
    "calc": "Sometimes", "mtrans": "Public_Transportation"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

LOGO_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQf1WWz9kmuLdJe3n7PciILEnlXyu10z7lpww&s"
ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def render_header():
    """Persistent header with ECC logo + app name + GR 27 badge."""
    st.markdown(f"""
    <div class="app-header animate-in">
        <img src="{LOGO_URL}" alt="ECC Logo">
        <span class="header-title">Obesity Risk Prediction</span>
        <span class="header-group">GR 27</span>
    </div>
    """, unsafe_allow_html=True)


def render_progress(current_step):
    """Numbered step indicators with connecting lines."""
    labels = ["Profil", "Nutrition", "Lifestyle"]
    html = '<div class="progress-container">'
    for i, label in enumerate(labels, 1):
        if i < current_step:
            c_cls, l_cls = "done", "done"
        elif i == current_step:
            c_cls, l_cls = "active", "active"
        else:
            c_cls, l_cls = "", ""

        icon = "✓" if i < current_step else str(i)

        html += '<div class="progress-step">'
        if i < len(labels):
            line_cls = "progress-line done" if i < current_step else "progress-line"
            html += f'<div class="{line_cls}"></div>'
        html += f'<div class="progress-circle {c_cls}">{icon}</div>'
        html += f'<div class="progress-label {l_cls}">{label}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_step_image(filename, alt="step illustration"):
    """Render a step image with constrained size."""
    path = os.path.join(ASSETS, filename)
    if os.path.exists(path):
        import base64
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="step-image-container animate-in">
            <img src="data:image/png;base64,{data}" alt="{alt}">
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL + PREPROCESSING (cached)
# --------------------------------------------------

DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_preprocessing():
    df = fetch_dataset()
    X_train, X_test, y_train, y_test, transformer, feature_names = preprocess_data(df)
    return transformer, feature_names

model = load_model()
transformer, feature_names = load_preprocessing()


# ====================================================
# STEP 0 — HOME
# ====================================================

if st.session_state.step == 0:

    st.markdown(f"""
    <div class="home-card animate-in">
        <div class="home-logo">
            <img src="{LOGO_URL}" alt="École Centrale Casablanca">
        </div>
        <h1 class="home-title">Obesity Risk<br>Prediction</h1>
        <div class="home-badge">GR 27</div>
        <p class="home-desc">
            Diagnostic intelligent du risque d'obésité propulsé par
            l'intelligence artificielle. 
            Répondez à quelques questions
            pour obtenir votre évaluation personnalisée.
        </p>
        <div class="home-features">
            <div class="feature-chip">📋 Profil biométrique</div>
            <div class="feature-chip">🥗 Habitudes alimentaires</div>
            <div class="feature-chip">🏃 Style de vie</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button("🚀  Commencer le diagnostic", width="stretch"):
            st.session_state.step = 1
            st.rerun()


# ====================================================
# STEP 1 — PROFIL
# ====================================================

elif st.session_state.step == 1:

    render_header()
    render_progress(1)

    st.markdown("""
    <div class="step-header animate-in">
        <div class="step-emoji">📋</div>
        <div class="step-title">Profil Biométrique</div>
        <div class="step-subtitle">Entrez vos informations de base</div>
    </div>
    """, unsafe_allow_html=True)

    render_step_image("step1_biometric.png", "Biometric profile illustration")

    st.session_state.age = st.slider("Âge", 10, 80, st.session_state.age)
    st.session_state.height = st.number_input("Taille (m)", 1.0, 2.2, st.session_state.height)
    st.session_state.weight = st.number_input("Poids (kg)", 30, 200, st.session_state.weight)
    st.session_state.gender = st.selectbox("Genre", ["Male", "Female"],
                                           index=["Male", "Female"].index(st.session_state.gender))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🏠 Accueil"):
            st.session_state.step = 0
            st.rerun()
    with col_b:
        if st.button("Suivant ➡️", width="stretch"):
            st.session_state.step = 2
            st.rerun()


# ====================================================
# STEP 2 — NUTRITION
# ====================================================

elif st.session_state.step == 2:

    render_header()
    render_progress(2)

    st.markdown("""
    <div class="step-header animate-in">
        <div class="step-emoji">🥗</div>
        <div class="step-title">Habitudes Alimentaires</div>
        <div class="step-subtitle">Décrivez votre alimentation quotidienne</div>
    </div>
    """, unsafe_allow_html=True)

    render_step_image("step2_eating.png", "Nutrition illustration")

    st.session_state.family_history = st.selectbox(
        "Antécédents familiaux de surpoids", ["yes", "no"],
        index=["yes", "no"].index(st.session_state.family_history))

    st.session_state.favc = st.selectbox(
        "Consommation d'aliments riches en calories", ["yes", "no"],
        index=["yes", "no"].index(st.session_state.favc))

    st.session_state.fcvc = st.slider("Consommation de légumes", 1.0, 3.0, st.session_state.fcvc)
    st.session_state.ncp = st.slider("Nombre de repas principaux", 1.0, 4.0, st.session_state.ncp)
    st.session_state.ch2o = st.slider("Consommation d'eau (L/jour)", 1.0, 3.0, st.session_state.ch2o)

    st.session_state.caec = st.selectbox(
        "Grignotage entre les repas",
        ["no", "Sometimes", "Frequently", "Always"],
        index=["no", "Sometimes", "Frequently", "Always"].index(st.session_state.caec))

    st.session_state.scc = st.selectbox(
        "Surveillance des calories", ["yes", "no"],
        index=["yes", "no"].index(st.session_state.scc))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅️ Retour", width="stretch"):
            st.session_state.step = 1
            st.rerun()
    with col_b:
        if st.button("Suivant ➡️", width="stretch"):
            st.session_state.step = 3
            st.rerun()


# ====================================================
# STEP 3 — LIFESTYLE
# ====================================================

elif st.session_state.step == 3:

    render_header()
    render_progress(3)

    st.markdown("""
    <div class="step-header animate-in">
        <div class="step-emoji">🏃</div>
        <div class="step-title">Style de Vie</div>
        <div class="step-subtitle">Parlez-nous de votre mode de vie</div>
    </div>
    """, unsafe_allow_html=True)

    render_step_image("step3_lifestyle.png", "Lifestyle illustration")

    st.session_state.faf = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, st.session_state.faf)
    st.session_state.tue = st.slider("Utilisation de la technologie (h/jour)", 0.0, 2.0, st.session_state.tue)

    st.session_state.smoke = st.selectbox(
        "Tabagisme", ["yes", "no"],
        index=["yes", "no"].index(st.session_state.smoke))

    st.session_state.calc = st.selectbox(
        "Consommation d'alcool",
        ["no", "Sometimes", "Frequently", "Always"],
        index=["no", "Sometimes", "Frequently", "Always"].index(st.session_state.calc))

    st.session_state.mtrans = st.selectbox(
        "Mode de transport",
        ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"],
        index=["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"].index(st.session_state.mtrans))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅️ Retour", width="stretch"):
            st.session_state.step = 2
            st.rerun()
    with col_b:
        if st.button("🔬 Lancer le diagnostic", width="stretch"):
            st.session_state.step = 4
            st.rerun()


# ====================================================
# STEP 4 — RESULTS
# ====================================================

elif st.session_state.step == 4:

    render_header()

    # ---- Build input dataframe ----
    input_df = pd.DataFrame({
        "Age": [st.session_state.age],
        "Height": [st.session_state.height],
        "Weight": [st.session_state.weight],
        "Gender": [st.session_state.gender],
        "family_history_with_overweight": [st.session_state.family_history],
        "FAVC": [st.session_state.favc],
        "FCVC": [st.session_state.fcvc],
        "NCP": [st.session_state.ncp],
        "CAEC": [st.session_state.caec],
        "SMOKE": [st.session_state.smoke],
        "CH2O": [st.session_state.ch2o],
        "SCC": [st.session_state.scc],
        "FAF": [st.session_state.faf],
        "TUE": [st.session_state.tue],
        "CALC": [st.session_state.calc],
        "MTRANS": [st.session_state.mtrans]
    })

    input_df = input_df[transformer.feature_names_in_]
    X_processed = transformer.transform(input_df)
    X = pd.DataFrame(X_processed, columns=feature_names)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = probabilities.max()
    bmi = st.session_state.weight / st.session_state.height ** 2

    # ---- Title ----
    st.markdown("""
    <div class="step-header animate-in">
        <div class="step-emoji">📊</div>
        <div class="step-title">Résultats du Diagnostic</div>
        <div class="step-subtitle">Votre évaluation personnalisée</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Summary cards ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Classe prédite", prediction)
    with c2:
        st.metric("Confiance", f"{confidence:.1%}")
    with c3:
        if bmi < 18.5:
            st.info(f"IMC : {bmi:.1f} — Sous-poids")
        elif bmi < 25:
            st.success(f"IMC : {bmi:.1f} — Poids sain")
        elif bmi < 30:
            st.warning(f"IMC : {bmi:.1f} — Surpoids")
        else:
            st.error(f"IMC : {bmi:.1f} — Obésité")

    # ---- BMI Gauge ----
    st.markdown('<div class="result-section">🎯 Indice de Masse Corporelle</div>', unsafe_allow_html=True)

    fig_bmi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={"text": "IMC", "font": {"family": "Outfit", "color": "#1e293b"}},
        number={"font": {"color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": "#94a3b8"},
            "bar": {"color": "#8b5cf6"},
            "steps": [
                {"range": [0, 18.5], "color": "#bfdbfe"},
                {"range": [18.5, 25], "color": "#bbf7d0"},
                {"range": [25, 30], "color": "#fef08a"},
                {"range": [30, 50], "color": "#fecaca"},
            ]
        }
    ))
    fig_bmi.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b", height=280)
    st.plotly_chart(fig_bmi, width="stretch")

    # ---- Lifestyle Radar ----
    st.markdown('<div class="result-section">🕸️ Profil de Mode de Vie</div>', unsafe_allow_html=True)

    categories = ["Légumes", "Repas", "Eau", "Activité physique", "Technologie"]
    values = [st.session_state.fcvc, st.session_state.ncp,
              st.session_state.ch2o, st.session_state.faf,
              st.session_state.tue]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(139, 92, 246, 0.15)',
        line_color='#8b5cf6',
        marker_color='#8b5cf6'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4], gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b",
        height=320
    )
    st.plotly_chart(fig_radar, width="stretch")

    # ---- Probability Distribution ----
    st.markdown('<div class="result-section">📈 Distribution des Probabilités</div>', unsafe_allow_html=True)

    bar_colors = []
    for p in probabilities:
        if p == confidence:
            bar_colors.append('#8b5cf6')
        else:
            bar_colors.append('#cbd5e1')

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=model.classes_, y=probabilities,
        marker_color=bar_colors,
        marker_line_width=0
    ))
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b",
        height=320,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#e2e8f0")
    )
    st.plotly_chart(fig_prob, width="stretch")

    # ---- SHAP ----
    st.markdown('<div class="result-section">🔍 Explicabilité du Modèle (SHAP)</div>', unsafe_allow_html=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    class_idx = np.argmax(probabilities)

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values.values[:, :, class_idx], X, show=False)
    st.pyplot(fig1)

    st.markdown('<div class="result-section">💧 Explication Locale</div>', unsafe_allow_html=True)

    explanation = shap_values[0, :, class_idx]
    fig2, ax2 = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig2)

    # ---- Footer ----
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button("🏠  Recommencer", width="stretch"):
            st.session_state.step = 0
            st.rerun()

    st.markdown("""
    <div class="app-footer">
        AI Obesity Risk Prediction — <strong>GR 27</strong> — École Centrale Casablanca<br>
        Modèle ML : CatBoost &nbsp;|&nbsp; Explicabilité : SHAP
    </div>
    """, unsafe_allow_html=True)

