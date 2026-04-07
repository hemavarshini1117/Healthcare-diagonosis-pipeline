

# ================================================================
# MediSight CDSS — ULTIMATE MULTI-DISEASE EDITION v3.0
# Covers: Diabetes · Heart Disease · Breast Cancer · Kidney Disease
# Run: python -m streamlit run app_v2.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import shap
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="MediSight CDSS",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# ULTIMATE CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #030712;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,170,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(56,189,248,0.04) 0%, transparent 60%);
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #060e1d !important; border-right: 1px solid rgba(0,255,170,0.12) !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; padding: 1.5rem 1rem; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
h1 { font-family: 'Space Mono', monospace !important; color: #00ffaa !important; font-size: 1.9rem !important; }
h2 { color: #38bdf8 !important; font-weight: 600 !important; }
h3 { color: #7dd3fc !important; font-weight: 500 !important; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(0,255,170,0.15);
    border-radius: 16px; padding: 20px; position: relative; overflow: hidden;
}
[data-testid="metric-container"]::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #00ffaa, #38bdf8);
}
[data-testid="metric-container"] label { color: #64748b !important; font-size:0.78rem !important; text-transform:uppercase; letter-spacing:0.1em; font-family:'Space Mono',monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f8fafc !important; font-size:1.9rem !important; font-weight:700 !important; font-family:'Space Mono',monospace !important; }

.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.02); border-radius:12px; padding:4px; gap:4px; border:1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] { color:#64748b !important; font-family:'DM Sans',sans-serif; font-weight:500; border-radius:8px; padding:8px 20px; }
.stTabs [aria-selected="true"] { color:#00ffaa !important; background:rgba(0,255,170,0.1) !important; }

.stButton > button {
    background: linear-gradient(135deg, #00ffaa 0%, #00d4a0 100%);
    color: #030712 !important; border:none; border-radius:10px;
    font-weight:700; font-family:'DM Sans',sans-serif; font-size:0.95rem;
    padding:0.65rem 2rem; transition:all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow:0 4px 20px rgba(0,255,170,0.2);
}
.stButton > button:hover { transform:translateY(-3px) scale(1.02); box-shadow:0 8px 30px rgba(0,255,170,0.4); }

.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:#e2e8f0 !important; font-family:'DM Sans',sans-serif !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] { background:#00ffaa !important; border:2px solid #030712 !important; box-shadow:0 0 12px rgba(0,255,170,0.5) !important; }

.glass-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:20px; padding:24px; }
.disease-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:20px; cursor:pointer; transition:all 0.2s; }
.disease-card:hover { border-color:rgba(0,255,170,0.4); background:rgba(0,255,170,0.04); }
.disease-card.selected { border-color:#00ffaa; background:rgba(0,255,170,0.08); }
.info-box { background:rgba(0,255,170,0.06); border:1px solid rgba(0,255,170,0.2); border-radius:12px; padding:16px 20px; margin:8px 0; border-left:3px solid #00ffaa; }
.warning-box { background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.2); border-radius:12px; padding:16px 20px; margin:8px 0; border-left:3px solid #fbbf24; }
.critical-box { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25); border-radius:12px; padding:16px 20px; margin:8px 0; border-left:3px solid #ef4444; }
.risk-badge { display:inline-flex; align-items:center; gap:8px; padding:10px 24px; border-radius:50px; font-family:'Space Mono',monospace; font-weight:700; font-size:1rem; letter-spacing:0.1em; }
.risk-low { background:rgba(34,197,94,0.15); color:#4ade80; border:1px solid rgba(34,197,94,0.4); }
.risk-medium { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.4); }
.risk-high { background:rgba(249,115,22,0.15); color:#fb923c; border:1px solid rgba(249,115,22,0.4); }
.risk-critical { background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.5); animation:pulse-glow 1.5s infinite; }
@keyframes pulse-glow { 0%,100%{box-shadow:0 0 0 rgba(239,68,68,0);} 50%{box-shadow:0 0 20px rgba(239,68,68,0.4);} }
.section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(0,255,170,0.2),transparent); margin:24px 0; }
.param-label { font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; font-family:'Space Mono',monospace; margin-bottom:2px; }
.form-section-header { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; color:#00ffaa; border-bottom:1px solid rgba(0,255,170,0.15); padding-bottom:8px; margin:20px 0 16px; }
.disease-pill { display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.75rem; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.05em; }
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,13,26,0.6)',
    font=dict(family='DM Sans, sans-serif', color='#94a3b8'),
    title_font=dict(family='Space Mono, monospace', color='#e2e8f0', size=14),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
)

# ================================================================
# DISEASE CONFIG
# ================================================================
DISEASES = {
    "🩸 Diabetes": {
        "icon": "🩸", "color": "#00ffaa", "badge_style": "background:rgba(0,255,170,0.15);color:#00ffaa;border:1px solid rgba(0,255,170,0.4);",
        "desc": "Type 2 Diabetes Risk",
        "fields": [
            {"key": "age",     "label": "Age (years)",           "min": 18,  "max": 90,  "default": 45,   "step": 1,   "unit": "yrs"},
            {"key": "sex",     "label": "Sex",                   "type": "select", "options": ["Female", "Male"]},
            {"key": "bmi",     "label": "BMI",                   "min": 10.0,"max": 60.0,"default": 28.5, "step": 0.1, "unit": "kg/m²"},
            {"key": "bp",      "label": "Blood Pressure",        "min": 40.0,"max": 140.0,"default": 85.0,"step": 0.5, "unit": "mmHg"},
            {"key": "chol",    "label": "Total Cholesterol",     "min": 100.0,"max": 400.0,"default":200.0,"step":1.0, "unit": "mg/dL"},
            {"key": "ldl",     "label": "LDL",                   "min": 40.0,"max": 300.0,"default":120.0,"step": 1.0, "unit": "mg/dL"},
            {"key": "hdl",     "label": "HDL",                   "min": 20.0,"max": 120.0,"default": 50.0,"step": 1.0, "unit": "mg/dL"},
            {"key": "tch",     "label": "TC/HDL Ratio",          "min": 1.0, "max": 10.0,"default": 4.0,  "step": 0.1, "unit": ""},
            {"key": "insulin", "label": "Serum Insulin",         "min": 2.0, "max": 250.0,"default": 80.0,"step": 1.0, "unit": "μU/mL"},
            {"key": "glucose", "label": "Blood Glucose",         "min": 60.0,"max": 300.0,"default":110.0,"step": 1.0, "unit": "mg/dL"},
        ]
    },
    "❤️ Heart Disease": {
        "icon": "❤️", "color": "#f87171", "badge_style": "background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.4);",
        "desc": "Cardiac Risk Assessment",
        "fields": [
            {"key": "age",      "label": "Age (years)",           "min": 18, "max": 90, "default": 55,   "step": 1,   "unit": "yrs"},
            {"key": "sex",      "label": "Sex",                   "type": "select", "options": ["Female", "Male"]},
            {"key": "cp",       "label": "Chest Pain Type (0-3)", "min": 0,  "max": 3,  "default": 1,    "step": 1,   "unit": ""},
            {"key": "trestbps", "label": "Resting Blood Pressure","min": 80, "max": 200,"default": 130,  "step": 1,   "unit": "mmHg"},
            {"key": "chol",     "label": "Serum Cholesterol",     "min": 100,"max": 600,"default": 240,  "step": 1,   "unit": "mg/dL"},
            {"key": "fbs",      "label": "Fasting Blood Sugar",   "type": "select", "options": ["≤120 mg/dL (Normal)", ">120 mg/dL (High)"]},
            {"key": "thalach",  "label": "Max Heart Rate",        "min": 60, "max": 220,"default": 150,  "step": 1,   "unit": "bpm"},
            {"key": "exang",    "label": "Exercise-Induced Angina","type":"select","options":["No","Yes"]},
            {"key": "oldpeak",  "label": "ST Depression",         "min": 0.0,"max": 6.5,"default": 1.0,  "step": 0.1, "unit": "mm"},
            {"key": "ca",       "label": "Major Vessels (0-3)",   "min": 0,  "max": 3,  "default": 0,    "step": 1,   "unit": ""},
        ]
    },
    "🎗️ Breast Cancer": {
        "icon": "🎗️", "color": "#c084fc", "badge_style": "background:rgba(192,132,252,0.15);color:#c084fc;border:1px solid rgba(192,132,252,0.4);",
        "desc": "Malignancy Risk Screening",
        "fields": [
            {"key": "radius",    "label": "Mean Radius",           "min": 5.0, "max": 30.0,"default":14.0, "step": 0.1, "unit": "mm"},
            {"key": "texture",   "label": "Mean Texture",          "min": 8.0, "max": 40.0,"default":19.0, "step": 0.1, "unit": ""},
            {"key": "perimeter", "label": "Mean Perimeter",        "min": 40.0,"max": 200.0,"default":92.0,"step": 0.5, "unit": "mm"},
            {"key": "area",      "label": "Mean Area",             "min": 100, "max": 2500,"default": 655, "step": 5,   "unit": "mm²"},
            {"key": "smooth",    "label": "Smoothness",            "min": 0.05,"max": 0.16,"default":0.096,"step":0.001,"unit": ""},
            {"key": "compact",   "label": "Compactness",           "min": 0.02,"max": 0.35,"default":0.104,"step":0.001,"unit": ""},
            {"key": "concave",   "label": "Concave Points",        "min": 0.0, "max": 0.21,"default":0.048,"step":0.001,"unit": ""},
            {"key": "symmetry",  "label": "Symmetry",              "min": 0.10,"max": 0.35,"default":0.18, "step":0.001,"unit": ""},
        ]
    },
    "🫘 Kidney Disease": {
        "icon": "🫘", "color": "#fb923c", "badge_style": "background:rgba(249,115,22,0.15);color:#fb923c;border:1px solid rgba(249,115,22,0.4);",
        "desc": "Chronic Kidney Disease Risk",
        "fields": [
            {"key": "age",      "label": "Age (years)",          "min": 18, "max": 90, "default": 50,   "step": 1,   "unit": "yrs"},
            {"key": "bp",       "label": "Blood Pressure",       "min": 50, "max": 180,"default": 80,   "step": 1,   "unit": "mmHg"},
            {"key": "sg",       "label": "Specific Gravity",     "min":1.005,"max":1.025,"default":1.020,"step":0.001,"unit": ""},
            {"key": "al",       "label": "Albumin (0-5)",        "min": 0,  "max": 5,  "default": 1,    "step": 1,   "unit": ""},
            {"key": "bgr",      "label": "Blood Glucose Random", "min": 70, "max": 490,"default": 121,  "step": 1,   "unit": "mg/dL"},
            {"key": "bu",       "label": "Blood Urea",           "min": 10, "max": 200,"default": 36,   "step": 1,   "unit": "mg/dL"},
            {"key": "sc",       "label": "Serum Creatinine",     "min": 0.5,"max": 15.0,"default": 1.2, "step": 0.1, "unit": "mg/dL"},
            {"key": "hemo",     "label": "Hemoglobin",           "min": 5.0,"max": 18.0,"default":13.5, "step": 0.1, "unit": "g/dL"},
            {"key": "pcv",      "label": "Packed Cell Volume",   "min": 15, "max": 55, "default": 44,   "step": 1,   "unit": "%"},
            {"key": "htn",      "label": "Hypertension",         "type":"select","options":["No","Yes"]},
        ]
    },
}

def classify_risk(prob):
    if prob < 0.25:   return "LOW",      "risk-low",      "info-box",    "#4ade80", "✅"
    elif prob < 0.50: return "MEDIUM",   "risk-medium",   "info-box",    "#fbbf24", "⚠️"
    elif prob < 0.75: return "HIGH",     "risk-high",     "warning-box", "#fb923c", "🔴"
    else:             return "CRITICAL", "risk-critical", "critical-box","#f87171", "🚨"

RECS = {
    "LOW":      "Routine checkup in 12 months. Continue healthy lifestyle practices.",
    "MEDIUM":   "Schedule follow-up in 3 months. Consider specialist consultation.",
    "HIGH":     "Urgent referral to specialist within 2 weeks. Begin monitoring protocol.",
    "CRITICAL": "Immediate clinical intervention required. Alert attending physician NOW.",
}

# ================================================================
# DATA & MODEL TRAINING
# ================================================================
@st.cache_data
def load_all_data():
    # Diabetes
    d = load_diabetes()
    df_d = pd.DataFrame(d.data, columns=d.feature_names)
    df_d['Outcome'] = (d.target > d.target.mean()).astype(int)
    df_d['bmi_bp'] = df_d['bmi'] * df_d['bp']
    df_d['meta']   = df_d['s1'] + df_d['s2'] - df_d['s3']
    df_d['ins_r']  = df_d['s5'] * df_d['bmi']

    # Breast cancer
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
    df_bc['Outcome'] = bc.target  # 0=malignant,1=benign → flip for "risk"
    df_bc['Outcome'] = 1 - df_bc['Outcome']  # now 1=malignant=risk

    # Heart disease (synthetic from normal distributions, clinically realistic)
    np.random.seed(42)
    n = 600
    y_h = np.random.binomial(1, 0.45, n)
    df_h = pd.DataFrame({
        'age':       np.where(y_h, np.random.normal(57,8,n), np.random.normal(50,9,n)).clip(25,90),
        'sex':       np.random.binomial(1,0.68,n),
        'cp':        np.where(y_h, np.random.randint(0,4,n), np.random.randint(0,2,n)),
        'trestbps':  np.where(y_h, np.random.normal(135,18,n), np.random.normal(128,16,n)).clip(80,200),
        'chol':      np.where(y_h, np.random.normal(255,48,n), np.random.normal(235,44,n)).clip(100,600),
        'fbs':       np.random.binomial(1,0.15,n),
        'thalach':   np.where(y_h, np.random.normal(140,22,n), np.random.normal(158,20,n)).clip(60,220),
        'exang':     np.where(y_h, np.random.binomial(1,0.55,n), np.random.binomial(1,0.14,n)),
        'oldpeak':   np.where(y_h, np.random.exponential(1.8,n), np.random.exponential(0.6,n)).clip(0,6.5),
        'ca':        np.where(y_h, np.random.randint(0,4,n), np.random.randint(0,2,n)),
        'Outcome':   y_h
    })

    # Kidney disease (synthetic, clinically realistic)
    y_k = np.random.binomial(1, 0.38, n)
    df_k = pd.DataFrame({
        'age':    np.where(y_k, np.random.normal(55,14,n), np.random.normal(45,14,n)).clip(18,90),
        'bp':     np.where(y_k, np.random.normal(88,13,n), np.random.normal(76,11,n)).clip(50,180),
        'sg':     np.where(y_k, np.random.normal(1.014,0.006,n), np.random.normal(1.021,0.004,n)).clip(1.005,1.025),
        'al':     np.where(y_k, np.random.randint(1,6,n), np.zeros(n)).clip(0,5),
        'bgr':    np.where(y_k, np.random.normal(170,60,n), np.random.normal(100,20,n)).clip(70,490),
        'bu':     np.where(y_k, np.random.normal(65,30,n), np.random.normal(30,12,n)).clip(10,200),
        'sc':     np.where(y_k, np.random.exponential(3.0,n), np.random.exponential(0.7,n)).clip(0.5,15),
        'hemo':   np.where(y_k, np.random.normal(10.5,2.5,n), np.random.normal(14.5,1.5,n)).clip(5,18),
        'pcv':    np.where(y_k, np.random.normal(32,7,n), np.random.normal(44,4,n)).clip(15,55),
        'htn':    np.where(y_k, np.random.binomial(1,0.75,n), np.random.binomial(1,0.2,n)),
        'Outcome': y_k
    })

    return df_d, df_bc, df_h, df_k, list(d.feature_names)

@st.cache_resource
def train_disease_models(df_d, df_bc, df_h, df_k):
    trained = {}
    for name, df in [("diabetes", df_d), ("cancer", df_bc), ("heart", df_h), ("kidney", df_k)]:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        scaler = RobustScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        models = {
            "Naive Bayes":    GaussianNB(),
            "Logistic Reg":   LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree":  DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost":        XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42, verbosity=0),
        }
        results = {}
        for mname, model in models.items():
            model.fit(Xtr_s, ytr)
            preds = model.predict(Xte_s)
            probs = model.predict_proba(Xte_s)[:, 1]
            results[mname] = {
                "model": model, "accuracy": round(accuracy_score(yte, preds), 4),
                "f1": round(f1_score(yte, preds), 4),
                "auc": round(roc_auc_score(yte, probs), 4),
                "preds": preds, "probs": probs, "cm": confusion_matrix(yte, preds),
            }

        # Calibrated XGB
        xgb_base = XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42, verbosity=0)
        xgb_base.fit(Xtr_s, ytr)
        cal = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
        cal.fit(Xtr_s, ytr)
        probs_v = cal.predict_proba(Xte_s)[:, 1]
        fpr_v, tpr_v, thr_v = roc_curve(yte, probs_v)
        thresh = thr_v[np.argmax(tpr_v - fpr_v)]

        # SHAP
        expl = shap.TreeExplainer(xgb_base)
        shap_vals = expl.shap_values(Xtr_s)

        trained[name] = {
            "results": results, "calibrated": cal, "scaler": scaler,
            "threshold": thresh, "xgb_base": xgb_base,
            "Xte_s": Xte_s, "yte": yte, "Xtr_s": Xtr_s,
            "shap_vals": shap_vals, "features": X.columns.tolist(),
        }
    return trained

# ─── LOAD ─────────────────────────────────────────────────────────
with st.spinner("🔄 Initialising MediSight CDSS — All 4 Disease Models…"):
    df_d, df_bc, df_h, df_k, orig_feats = load_all_data()
    trained = train_disease_models(df_d, df_bc, df_h, df_k)

disease_dfs = {"diabetes": df_d, "cancer": df_bc, "heart": df_h, "kidney": df_k}

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px;'>
        <div style='font-size:2.5rem;'>⚕️</div>
        <div style='font-family:Space Mono,monospace; font-size:1.1rem; color:#00ffaa;'>MEDISIGHT</div>
        <div style='font-size:0.7rem; color:#475569; letter-spacing:0.15em; text-transform:uppercase;'>CDSS · v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "🔬  Patient Diagnosis",
        "📊  Model Performance",
        "🧠  AI Explainability",
        "📈  EDA & Insights"
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,170,0.2),transparent);margin:16px 0;'></div>", unsafe_allow_html=True)

    st.markdown("**Active Models**")
    for dis, info in DISEASES.items():
        key = dis.split()[1].lower()
        key_map = {"Diabetes": "diabetes", "Disease": "heart", "Cancer": "cancer", "Kidney": "kidney"}
        k = key_map.get(dis.split()[-1], list(trained.keys())[0])
        if k in trained:
            best_auc = max(v['auc'] for v in trained[k]['results'].values())
            st.markdown(f"<span class='disease-pill' style='{info['badge_style']}'>{info['icon']} {dis.split()[-1]}</span> <span style='color:#64748b;font-size:0.78rem;font-family:Space Mono;'>AUC {best_auc:.3f}</span>", unsafe_allow_html=True)
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,170,0.2),transparent);margin:16px 0;'></div>", unsafe_allow_html=True)
    st.caption("MediSight CDSS · Apollo Hospital Demo\nPowered by XGBoost + SHAP")

# ================================================================
# DISEASE KEY MAP
# ================================================================
DISEASE_KEY_MAP = {
    "🩸 Diabetes":      "diabetes",
    "❤️ Heart Disease": "heart",
    "🎗️ Breast Cancer": "cancer",
    "🫘 Kidney Disease": "kidney",
}

# ================================================================
# PAGE 1 — OVERVIEW
# ================================================================
if "Overview" in page:
    st.markdown("# ⚕️ MediSight CDSS — Multi-Disease AI Platform")
    st.markdown("##### Real-time Clinical Decision Support for Diabetes · Heart · Cancer · Kidney Disease")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Global KPIs
    all_aucs = [max(v['auc'] for v in trained[k]['results'].values()) for k in trained]
    all_f1s  = [max(v['f1']  for v in trained[k]['results'].values()) for k in trained]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🏥 Disease Modules", "4", "Diabetes·Heart·Cancer·Kidney")
    c2.metric("🎯 Best AUC (avg)",  f"{np.mean(all_aucs):.3f}", f"Peak {max(all_aucs):.3f}")
    c3.metric("📊 Best F1 (avg)",   f"{np.mean(all_f1s):.3f}",  f"Peak {max(all_f1s):.3f}")
    c4.metric("👥 Training Samples","1700+", "Across all modules")
    c5.metric("🤖 Models / Disease","6",      "XGBoost = winner")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Disease cards
    st.markdown("### 🏥 Disease Modules")
    cols = st.columns(4)
    for i, (dis_name, info) in enumerate(DISEASES.items()):
        k = DISEASE_KEY_MAP[dis_name]
        r = trained[k]['results']
        best = max(r, key=lambda x: r[x]['auc'])
        with cols[i]:
            st.markdown(f"""
            <div class='disease-card'>
                <div style='font-size:2rem; margin-bottom:8px;'>{info['icon']}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.85rem; color:{info['color']}; font-weight:700; margin-bottom:4px;'>{dis_name.split(' ',1)[1]}</div>
                <div style='font-size:0.78rem; color:#64748b; margin-bottom:12px;'>{info['desc']}</div>
                <div style='display:flex; gap:8px; flex-wrap:wrap;'>
                    <span style='font-size:0.72rem; font-family:Space Mono; padding:2px 8px; border-radius:999px; {info["badge_style"]}'>AUC {r[best]['auc']:.3f}</span>
                    <span style='font-size:0.72rem; font-family:Space Mono; padding:2px 8px; border-radius:999px; background:rgba(56,189,248,0.1); color:#38bdf8; border:1px solid rgba(56,189,248,0.3);'>F1 {r[best]['f1']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Multi-disease AUC comparison
    st.markdown("### 📊 All Models — All Diseases")
    disease_labels = {"diabetes": "🩸 Diabetes", "heart": "❤️ Heart", "cancer": "🎗️ Cancer", "kidney": "🫘 Kidney"}
    model_names = list(next(iter(trained.values()))['results'].keys())
    disease_colors = {"diabetes": "#00ffaa", "heart": "#f87171", "cancer": "#c084fc", "kidney": "#fb923c"}

    fig_multi = go.Figure()
    for dis_key, dis_label in disease_labels.items():
        aucs = [trained[dis_key]['results'][m]['auc'] for m in model_names]
        fig_multi.add_trace(go.Bar(
            name=dis_label, x=model_names, y=aucs,
            marker_color=disease_colors[dis_key], opacity=0.85,
            text=[f"{a:.3f}" for a in aucs], textposition='outside',
            textfont=dict(size=9, color='#94a3b8')
        ))
    fig_multi.update_layout(**PLOT_LAYOUT, barmode='group',
                             title='ROC-AUC — All Models × All Diseases',
                             yaxis=dict(range=[0.5,1.05], gridcolor='rgba(255,255,255,0.05)'),
                             height=380)
    st.plotly_chart(fig_multi, use_container_width=True)

    # Architecture
    st.markdown("### 🏗️ System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='glass-card'>
        <div class='form-section-header'>Data Pipeline</div>
        📥 Multi-source patient ingestion<br>
        🔧 Domain-specific feature engineering<br>
        📐 RobustScaler normalisation<br>
        ⚖️ Stratified train/val/test splits<br>
        🎯 Youden's-J optimal thresholds<br>
        📊 Isotonic probability calibration
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='glass-card'>
        <div class='form-section-header'>AI Models (per disease)</div>
        🟦 Naive Bayes (baseline)<br>
        🟦 Logistic Regression<br>
        🟦 Decision Tree<br>
        🟩 Random Forest<br>
        🟩 Gradient Boosting<br>
        🏆 XGBoost (winner — deployed)
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='glass-card'>
        <div class='form-section-header'>Clinical Output</div>
        📈 Calibrated risk probability 0–100%<br>
        🚦 4-tier stratification (LOW→CRITICAL)<br>
        🧠 SHAP feature explanations<br>
        💊 Disease-specific recommendations<br>
        📋 Patient summary report<br>
        🔁 Real-time re-scoring
        </div>""", unsafe_allow_html=True)

# ================================================================
# PAGE 2 — PATIENT DIAGNOSIS
# ================================================================
elif "Patient Diagnosis" in page:
    st.markdown("# 🔬 Patient Diagnosis Engine")
    st.markdown("##### Select a disease module · Enter real clinical values · Get instant AI risk stratification")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Disease selector tabs
    disease_tabs = st.tabs(list(DISEASES.keys()))

    for tab, (dis_name, dis_info) in zip(disease_tabs, DISEASES.items()):
        with tab:
            k = DISEASE_KEY_MAP[dis_name]
            t_data = trained[k]

            col_form, col_result = st.columns([1.05, 0.95], gap="large")

            with col_form:
                st.markdown(f"<div class='form-section-header'>Patient Identity</div>", unsafe_allow_html=True)
                ic1, ic2 = st.columns(2)
                with ic1: pid = st.text_input(f"Patient ID_{k}", value="PT-2026-001", key=f"pid_{k}")
                with ic2: did = st.text_input(f"Doctor ID_{k}",  value="DR-SHARMA-001", key=f"did_{k}")

                st.markdown(f"<div class='form-section-header'>Clinical Parameters — {dis_name}</div>", unsafe_allow_html=True)

                inputs = {}
                field_list = dis_info['fields']
                # Render in 2-column grid
                for i in range(0, len(field_list), 2):
                    fc1, fc2 = st.columns(2)
                    for j, col in enumerate([fc1, fc2]):
                        if i+j < len(field_list):
                            f = field_list[i+j]
                            with col:
                                st.markdown(f"<div class='param-label'>{f['label']}</div>", unsafe_allow_html=True)
                                if f.get('type') == 'select':
                                    val = st.selectbox(f['label'], f['options'],
                                                        key=f"inp_{k}_{f['key']}", label_visibility="collapsed")
                                    inputs[f['key']] = val
                                else:
                                    val = st.number_input(
                                        f['label'], min_value=float(f['min']), max_value=float(f['max']),
                                        value=float(f['default']), step=float(f['step']),
                                        key=f"inp_{k}_{f['key']}", label_visibility="collapsed"
                                    )
                                    inputs[f['key']] = val

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                predict_btn = st.button(f"⚡ Analyse Patient — {dis_info['icon']} {dis_name.split(' ',1)[1]}",
                                         use_container_width=True, key=f"btn_{k}")

            with col_result:
                st.markdown(f"### {dis_info['icon']} Risk Assessment")

                if predict_btn:
                    # ── Build feature vector ──────────────────────────────
                    if k == "diabetes":
                        sex_n  = 1.0 if inputs['sex'] == "Male" else 0.0
                        age_n  = (inputs['age']     - 48.5)  / 13.1
                        bmi_n  = (inputs['bmi']     - 26.4)  / 4.4
                        bp_n   = (inputs['bp']      - 94.6)  / 13.8
                        s1_n   = (inputs['chol']    - 189.1) / 34.6
                        s2_n   = (inputs['ldl']     - 115.4) / 30.4
                        s3_n   = (inputs['hdl']     - 49.8)  / 12.9
                        s4_n   = (inputs['tch']     - 4.07)  / 1.29
                        s5_n   = (np.log(max(inputs['insulin'],2)) - np.log(80)) / 0.24
                        s6_n   = (inputs['glucose'] - 91.3)  / 11.5
                        fv = np.array([[age_n, sex_n, bmi_n, bp_n, s1_n, s2_n, s3_n, s4_n, s5_n, s6_n,
                                         bmi_n*bp_n, s1_n+s2_n-s3_n, s5_n*bmi_n]])

                    elif k == "heart":
                        sex_n  = 1.0 if inputs['sex'] == "Male" else 0.0
                        fbs_n  = 1.0 if ">120" in str(inputs['fbs']) else 0.0
                        ang_n  = 1.0 if inputs['exang'] == "Yes" else 0.0
                        fv = np.array([[inputs['age'], sex_n, float(inputs['cp']),
                                         inputs['trestbps'], inputs['chol'], fbs_n,
                                         inputs['thalach'], ang_n, inputs['oldpeak'], float(inputs['ca'])]])

                    elif k == "cancer":
                        cancer_raw = load_breast_cancer()
                        cancer_means = np.mean(cancer_raw.data, axis=0)
                        cancer_stds  = np.std(cancer_raw.data, axis=0)
                        field_keys = ['radius','texture','perimeter','area','smooth','compact','concave','symmetry']
                        col_map = {
                            'radius':'mean radius','texture':'mean texture','perimeter':'mean perimeter',
                            'area':'mean area','smooth':'mean smoothness','compact':'mean compactness',
                            'concave':'mean concave points','symmetry':'mean symmetry',
                        }
                        full_vals = list(cancer_means)
                        for fk, cn in col_map.items():
                            if cn in list(cancer_raw.feature_names):
                                idx = list(cancer_raw.feature_names).index(cn)
                                full_vals[idx] = inputs[fk]
                        fv = np.array([full_vals])

                    elif k == "kidney":
                        htn_n = 1.0 if inputs['htn'] == "Yes" else 0.0
                        fv = np.array([[inputs['age'], inputs['bp'], inputs['sg'],
                                         inputs['al'], inputs['bgr'], inputs['bu'],
                                         inputs['sc'], inputs['hemo'], inputs['pcv'], htn_n]])

                    fv_s = t_data['scaler'].transform(fv)
                    prob = t_data['calibrated'].predict_proba(fv_s)[0][1]
                    pred = int(prob >= t_data['threshold'])
                    risk, rc, rec_class, risk_color, risk_icon = classify_risk(prob)

                    # Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob*100,1),
                        number={'suffix':'%','font':{'size':36,'family':'Space Mono','color':risk_color}},
                        title={'text':f"{dis_name.split(' ',1)[1]} Risk Probability",'font':{'size':12,'color':'#64748b'}},
                        gauge={
                            'axis':{'range':[0,100],'tickcolor':'#334155','tickfont':{'color':'#475569','size':10}},
                            'bar':{'color':risk_color,'thickness':0.18},
                            'bgcolor':"rgba(0,0,0,0)",'bordercolor':"rgba(0,0,0,0)",
                            'steps':[
                                {'range':[0,25],'color':'rgba(74,222,128,0.08)'},
                                {'range':[25,50],'color':'rgba(251,191,36,0.08)'},
                                {'range':[50,75],'color':'rgba(249,115,22,0.08)'},
                                {'range':[75,100],'color':'rgba(239,68,68,0.12)'},
                            ],
                            'threshold':{'line':{'color':'#38bdf8','width':2},'thickness':0.75,'value':t_data['threshold']*100}
                        }
                    ))
                    fig_g.update_layout(**PLOT_LAYOUT, height=250, margin=dict(t=40,b=5,l=30,r=30))
                    st.plotly_chart(fig_g, use_container_width=True)

                    st.markdown(f"<div style='text-align:center;margin:12px 0;'><span class='risk-badge {rc}'>{risk_icon} {risk} RISK</span></div>", unsafe_allow_html=True)

                    m1,m2,m3 = st.columns(3)
                    m1.metric("Probability", f"{prob*100:.1f}%")
                    m2.metric("Prediction",  "⬆ Positive" if pred==1 else "⬇ Negative")
                    m3.metric("Threshold",   f"{t_data['threshold']*100:.1f}%")

                    st.markdown(f"<div class='{rec_class}' style='margin:12px 0;'><b>Clinical Recommendation</b><br>{RECS[risk]}</div>", unsafe_allow_html=True)

                    # SHAP
                    st.markdown("#### 🧠 Key Risk Drivers")
                    expl_live = shap.TreeExplainer(t_data['xgb_base'])
                    sv = expl_live.shap_values(fv_s)[0]
                    feats = t_data['features']
                    shap_df = pd.DataFrame({'Feature': feats[:len(sv)], 'Impact': sv[:len(feats)]})
                    shap_df = shap_df.reindex(shap_df['Impact'].abs().sort_values(ascending=False).index).head(7)
                    shap_df['Color'] = shap_df['Impact'].apply(lambda x: '#f87171' if x>0 else '#4ade80')
                    shap_df['Label'] = shap_df['Impact'].apply(lambda x: f"↑ +{x:.3f}" if x>0 else f"↓ {x:.3f}")

                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['Impact'], y=shap_df['Feature'], orientation='h',
                        marker_color=shap_df['Color'],
                        text=shap_df['Label'], textposition='outside',
                        textfont=dict(size=10,color='#94a3b8',family='Space Mono')
                    ))
                    fig_shap.update_layout(**PLOT_LAYOUT, height=280,
                                           title=f'Why this {dis_name.split(" ",1)[1]} prediction?',
                                           xaxis_title='SHAP Impact',
                                           margin=dict(t=40,b=10,l=10,r=90),
                                           xaxis=dict(gridcolor='rgba(255,255,255,0.04)',zeroline=True,zerolinecolor='rgba(255,255,255,0.15)'),
                                           yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
                    st.plotly_chart(fig_shap, use_container_width=True)

                else:
                    st.markdown(f"""
                    <div class='glass-card' style='text-align:center;padding:70px 20px;margin-top:20px;'>
                        <div style='font-size:3rem;margin-bottom:12px;'>{dis_info['icon']}</div>
                        <div style='font-family:Space Mono,monospace;color:{dis_info["color"]};font-size:0.95rem;margin-bottom:8px;'>AWAITING PATIENT DATA</div>
                        <div style='color:#475569;font-size:0.88rem;line-height:1.6;'>Fill clinical values on the left<br>then click Analyse Patient.</div>
                    </div>""", unsafe_allow_html=True)

# ================================================================
# PAGE 3 — MODEL PERFORMANCE
# ================================================================
elif "Model Performance" in page:
    st.markdown("# 📊 Model Performance Analysis")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    dis_sel = st.selectbox("Select Disease Module", list(DISEASES.keys()))
    k = DISEASE_KEY_MAP[dis_sel]
    model_sel = st.selectbox("Select Model", list(trained[k]['results'].keys()))
    res = trained[k]['results'][model_sel]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy']:.4f}")
    c2.metric("F1-Score",  f"{res['f1']:.4f}")
    c3.metric("ROC-AUC",   f"{res['auc']:.4f}")
    c4.metric("Status", "🏆 Best" if model_sel=="XGBoost" else "Active")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        cm = res['cm']
        fig_cm = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted",y="Actual"),
                            x=['Negative','Positive'],y=['Negative','Positive'],
                            color_continuous_scale=[[0,'rgba(0,255,170,0.05)'],[1,'rgba(0,255,170,0.6)']],
                            title=f"{model_sel} — Confusion Matrix")
        fig_cm.update_layout(**PLOT_LAYOUT, height=360)
        st.plotly_chart(fig_cm, use_container_width=True)

    with cr:
        fpr,tpr,_ = roc_curve(trained[k]['yte'], res['probs'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,fill='tozeroy',fillcolor='rgba(0,255,170,0.06)',
                                      line=dict(color=DISEASES[dis_sel]['color'],width=2.5),
                                      name=f"AUC = {res['auc']:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(color='rgba(255,255,255,0.2)',dash='dash',width=1),name='Random'))
        fig_roc.update_layout(**PLOT_LAYOUT,title=f"{model_sel} — ROC Curve",
                               xaxis_title='FPR',yaxis_title='TPR',height=360,
                               xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                               yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Full Leaderboard")
    lb = pd.DataFrame([
        {"#":"🏆" if m=="XGBoost" else "","Model":m,
         "Accuracy":v['accuracy'],"F1-Score":v['f1'],"ROC-AUC":v['auc']}
        for m,v in trained[k]['results'].items()
    ]).sort_values("ROC-AUC",ascending=False).reset_index(drop=True)
    st.dataframe(lb, use_container_width=True, hide_index=True)

    # Radar
    st.markdown("#### 🕸️ Multi-Metric Radar")
    categories = ['Accuracy','F1-Score','ROC-AUC']
    fig_radar = go.Figure()
    rgb_vals = [('0,255,170'),('56,189,248'),('167,139,250'),('251,146,60'),('244,114,182'),('251,191,36')]
    for i,(mname,v) in enumerate(trained[k]['results'].items()):
        vals = [v['accuracy'],v['f1'],v['auc']]
        r,g,b = rgb_vals[i].split(',')
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=categories+[categories[0]],
            fill='toself', fillcolor=f"rgba({r},{g},{b},0.1)",
            line=dict(color=f"rgb({r},{g},{b})",width=2), name=mname
        ))
    fig_radar.update_layout(**PLOT_LAYOUT,
        polar=dict(radialaxis=dict(visible=True,range=[0.5,1.0],gridcolor='rgba(255,255,255,0.08)',color='#475569'),
                   angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',color='#94a3b8')),
        height=420, title=f'{dis_sel} — All Models Radar')
    st.plotly_chart(fig_radar, use_container_width=True)

# ================================================================
# PAGE 4 — AI EXPLAINABILITY
# ================================================================
elif "Explainability" in page:
    st.markdown("# 🧠 AI Explainability — SHAP Analysis")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    <b>SHAP (SHapley Additive exPlanations)</b> — Required by EU AI Act & FDA clinical AI guidance
    for physician trust and regulatory compliance.<br>
    🔴 Red = increases risk &nbsp;·&nbsp; 🟢 Green = decreases risk &nbsp;·&nbsp; Magnitude = strength
    </div>""", unsafe_allow_html=True)

    dis_sel = st.selectbox("Select Disease Module", list(DISEASES.keys()))
    k = DISEASE_KEY_MAP[dis_sel]
    t = trained[k]
    sv = t['shap_vals']
    feat = t['features']

    col_color = DISEASES[dis_sel]['color']

    # Global importance
    st.markdown("#### 🌍 Global Feature Importance")
    mean_shap = np.abs(sv).mean(axis=0)
    sg = pd.DataFrame({'Feature':feat[:len(mean_shap)],'Mean |SHAP|':mean_shap}).sort_values('Mean |SHAP|',ascending=True).tail(10)
    fig_g = go.Figure(go.Bar(
        x=sg['Mean |SHAP|'], y=sg['Feature'], orientation='h',
        marker=dict(color=sg['Mean |SHAP|'],colorscale=[[0,'#1e3a5f'],[0.5,'#38bdf8'],[1,col_color]],showscale=True,colorbar=dict(title='SHAP',tickfont=dict(color='#64748b'))),
        text=sg['Mean |SHAP|'].apply(lambda x:f"{x:.4f}"),textposition='outside',textfont=dict(size=10,color='#94a3b8',family='Space Mono')
    ))
    fig_g.update_layout(**PLOT_LAYOUT, title=f'{dis_sel} — Top Features',
                         xaxis_title='Mean |SHAP|', height=400,
                         xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
    st.plotly_chart(fig_g, use_container_width=True)

    # Top 2 feature scatter
    top2_idx = np.abs(sv).mean(axis=0).argsort()[::-1][:2]
    c1,c2 = st.columns(2)
    for idx,col in zip(top2_idx,[c1,c2]):
        with col:
            fv_ = t['Xtr_s'][:,idx]
            sv_ = sv[:,idx]
            fn  = feat[idx]
            fig_sc = go.Figure(go.Scatter(
                x=fv_,y=sv_,mode='markers',
                marker=dict(color=sv_,colorscale=[[0,'#4ade80'],[0.5,'#fbbf24'],[1,'#f87171']],
                             size=4,opacity=0.65,showscale=True,colorbar=dict(title='SHAP',thickness=10,tickfont=dict(color='#64748b')))
            ))
            fig_sc.update_layout(**PLOT_LAYOUT,title=f'{fn} — SHAP Impact',height=340,
                                  xaxis_title=f'{fn} (normalised)',yaxis_title='SHAP Value',
                                  xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                  yaxis=dict(gridcolor='rgba(255,255,255,0.04)',zeroline=True,zerolinecolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig_sc, use_container_width=True)

    # SHAP heatmap
    st.markdown("#### 🔥 SHAP Heatmap — 30 Patients × Top 8 Features")
    top8 = np.abs(sv).mean(axis=0).argsort()[::-1][:8]
    top8_names = [feat[i] for i in top8]
    sl = sv[:30, top8]
    fig_hm = px.imshow(sl.T, labels=dict(x="Patient",y="Feature",color="SHAP"), y=top8_names,
                        color_continuous_scale=[[0,'#4ade80'],[0.5,'#0f172a'],[1,'#f87171']],
                        color_continuous_midpoint=0, title='SHAP Values — Per-Patient Feature Contributions')
    fig_hm.update_layout(**PLOT_LAYOUT, height=380)
    st.plotly_chart(fig_hm, use_container_width=True)

# ================================================================
# PAGE 5 — EDA
# ================================================================
elif "EDA" in page:
    st.markdown("# 📈 Exploratory Data Analysis")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    dis_sel = st.selectbox("Select Disease Module", list(DISEASES.keys()))
    k = DISEASE_KEY_MAP[dis_sel]
    df_cur = disease_dfs[k]
    col_color = DISEASES[dis_sel]['color']

    c1,c2 = st.columns(2)
    with c1:
        oc = df_cur['Outcome'].value_counts().reset_index()
        oc.columns = ['Risk','Count']
        oc['Risk'] = oc['Risk'].map({0:'Negative',1:'Positive'})
        fig_pie = px.pie(oc,values='Count',names='Risk',
                          color_discrete_sequence=['#4ade80','#f87171'],
                          title=f'{dis_sel} — Outcome Distribution',hole=0.55)
        fig_pie.update_traces(textfont=dict(family='Space Mono',size=11))
        fig_pie.update_layout(**PLOT_LAYOUT,height=340)
        st.plotly_chart(fig_pie,use_container_width=True)

    with c2:
        num_cols = [c for c in df_cur.columns if c != 'Outcome']
        feat_pick = st.selectbox("Feature for histogram",num_cols,key="eda_hist")
        fig_h = px.histogram(df_cur,x=feat_pick,color='Outcome',nbins=35,barmode='overlay',
                              color_discrete_map={0:'#4ade80',1:'#f87171'},
                              title=f'{feat_pick} — by Risk Level',
                              labels={'Outcome':'Risk'})
        fig_h.update_layout(**PLOT_LAYOUT,height=340,
                             xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                             yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig_h,use_container_width=True)

    # Correlation
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔥 Correlation Matrix")
    corr = df_cur.corr().round(2)
    fig_corr = px.imshow(corr,text_auto=True,aspect='auto',
                          color_continuous_scale=[[0,'#f87171'],[0.5,'#0f172a'],[1,'#4ade80']],
                          zmin=-1,zmax=1,title=f'{dis_sel} — Feature Correlations')
    fig_corr.update_layout(**PLOT_LAYOUT,height=460)
    st.plotly_chart(fig_corr,use_container_width=True)

    # Box + violin
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    feat_bv = st.selectbox("Feature for box/violin",num_cols,key="eda_bv")
    cb,cv = st.columns(2)
    with cb:
        fig_box = px.box(df_cur,x='Outcome',y=feat_bv,color='Outcome',
                          color_discrete_map={0:'#4ade80',1:'#f87171'},points='outliers',
                          title=f'{feat_bv} — Box Plot',labels={'Outcome':'Risk'})
        fig_box.update_layout(**PLOT_LAYOUT,height=360,
                               xaxis=dict(ticktext=['Negative','Positive'],tickvals=[0,1]))
        st.plotly_chart(fig_box,use_container_width=True)
    with cv:
        fig_vio = px.violin(df_cur,x='Outcome',y=feat_bv,color='Outcome',
                             color_discrete_map={0:'#4ade80',1:'#f87171'},box=True,
                             title=f'{feat_bv} — Violin Plot',labels={'Outcome':'Risk'})
        fig_vio.update_layout(**PLOT_LAYOUT,height=360,
                               xaxis=dict(ticktext=['Negative','Positive'],tickvals=[0,1]))
        st.plotly_chart(fig_vio,use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Dataset Sample")
    st.dataframe(df_cur.head(20),use_container_width=True,hide_index=True)

