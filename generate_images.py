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

PLOT_LAYOUT = dict(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,13,26,0.6)',
    font=dict(family='DM Sans, sans-serif', color='#94a3b8'),
    title_font=dict(family='Space Mono, monospace', color='#e2e8f0', size=14),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
)

def classify_risk(prob):
    if prob < 0.25:   return "LOW",      "risk-low",      "info-box",    "#4ade80", "✅"
    elif prob < 0.50: return "MEDIUM",   "risk-medium",   "info-box",    "#fbbf24", "⚠️"
    elif prob < 0.75: return "HIGH",     "risk-high",     "warning-box", "#fb923c", "🔴"
    else:             return "CRITICAL", "risk-critical", "critical-box","#f87171", "🚨"

# Load data and train models (same as app)
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

# Load
df_d, df_bc, df_h, df_k, orig_feats = load_all_data()
trained = train_disease_models(df_d, df_bc, df_h, df_k)

# Generate multi AUC plot
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
fig_multi.write_image("multi_auc.png")

# Generate sample gauge for diabetes
k = "diabetes"
t_data = trained[k]
# Sample input
inputs = {'age': 45, 'sex': 'Male', 'bmi': 28.5, 'bp': 85, 'chol': 200, 'ldl': 120, 'hdl': 50, 'tch': 4.0, 'insulin': 80, 'glucose': 110}
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
fv_s = t_data['scaler'].transform(fv)
prob = t_data['calibrated'].predict_proba(fv_s)[0][1]
risk, rc, rec_class, risk_color, risk_icon = classify_risk(prob)

fig_g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(prob*100,1),
    number={'suffix':'%','font':{'size':36,'family':'Space Mono','color':risk_color}},
    title={'text':'Diabetes Risk Probability','font':{'size':12,'color':'#64748b'}},
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
fig_g.write_image("gauge_diabetes.png")

print("Output images generated: multi_auc.png and gauge_diabetes.png")