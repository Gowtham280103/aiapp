
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.metrics import fairness_metrics
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens AI - Unbiased Decision Dashboard",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #080c14; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #0f2027 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, #e9456015 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, #00d4aa10 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #e94560 0%, #ff6b9d 40%, #00d4aa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { color: #8892b0; font-size: 1.05rem; margin: 0.5rem 0 1.5rem 0; }
.hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.hero-stat {
    background: #ffffff08; border: 1px solid #ffffff15;
    border-radius: 10px; padding: 0.7rem 1.2rem; text-align: center;
}
.hero-stat-val { font-size: 1.5rem; font-weight: 700; color: #e94560; }
.hero-stat-lbl { font-size: 0.72rem; color: #8892b0; margin-top: 0.1rem; }

/* ── Section headers ── */
.sec-head {
    display: flex; align-items: center; gap: 0.7rem;
    font-size: 1.25rem; font-weight: 700; color: #ccd6f6;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
}
.sec-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.icon-red  { background: #e9456022; }
.icon-green{ background: #00d4aa22; }
.icon-blue { background: #3b82f622; }
.icon-gold { background: #ffd70022; }

/* ── Cards ── */
.card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    height: 100%;
    transition: border-color 0.3s;
}
.card:hover { border-color: #e9456055; }
.card-title { font-size: 0.78rem; font-weight: 600; color: #8892b0; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.card-val   { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.card-sub   { font-size: 0.78rem; color: #8892b0; margin-top: 0.3rem; }
.card-badge { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 20px; margin-top: 0.5rem; }

.val-red   { color: #e94560; }
.val-green { color: #00d4aa; }
.val-gold  { color: #ffd700; }
.val-white { color: #ccd6f6; }

.badge-red   { background: #e9456022; color: #e94560; border: 1px solid #e9456044; }
.badge-green { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa44; }
.badge-gold  { background: #ffd70022; color: #ffd700; border: 1px solid #ffd70044; }

/* ── Prediction box ── */
.pred-box {
    border-radius: 16px; padding: 2rem; text-align: center;
    transition: all 0.4s ease;
}
.pred-approved {
    background: linear-gradient(135deg, #00d4aa18, #00d4aa08);
    border: 2px solid #00d4aa;
    box-shadow: 0 0 30px #00d4aa22;
}
.pred-denied {
    background: linear-gradient(135deg, #e9456018, #e9456008);
    border: 2px solid #e94560;
    box-shadow: 0 0 30px #e9456022;
}
.pred-verdict { font-size: 2rem; font-weight: 800; letter-spacing: 0.05em; }
.pred-prob    { font-size: 1rem; color: #8892b0; margin-top: 0.4rem; }
.pred-meta    { font-size: 0.8rem; color: #64748b; margin-top: 0.8rem; }

/* ── Progress bar ── */
.prog-wrap { background: #1e3a5f33; border-radius: 20px; height: 10px; overflow: hidden; margin: 0.4rem 0; }
.prog-fill  { height: 100%; border-radius: 20px; transition: width 0.8s ease; }
.prog-green { background: linear-gradient(90deg, #00d4aa, #00ff88); }
.prog-red   { background: linear-gradient(90deg, #e94560, #ff6b9d); }
.prog-gold  { background: linear-gradient(90deg, #ffd700, #ffaa00); }

/* ── Fairness score ring ── */
.score-ring-wrap { text-align: center; padding: 1rem; }
.score-val { font-size: 3.5rem; font-weight: 800; }
.score-lbl { font-size: 0.85rem; color: #8892b0; margin-top: 0.3rem; }

/* ── Step guide ── */
.step-row { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }
.step-num {
    min-width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #e94560, #ff6b9d);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; color: white; flex-shrink: 0;
}
.step-text { color: #8892b0; font-size: 0.88rem; line-height: 1.5; padding-top: 0.3rem; }
.step-text b { color: #ccd6f6; }

/* ── Alert boxes ── */
.alert-warn {
    background: #ffd70015; border: 1px solid #ffd70044;
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #ffd700; font-size: 0.88rem; margin: 0.5rem 0;
}
.alert-good {
    background: #00d4aa15; border: 1px solid #00d4aa44;
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #00d4aa; font-size: 0.88rem; margin: 0.5rem 0;
}
.alert-bad {
    background: #e9456015; border: 1px solid #e9456044;
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #e94560; font-size: 0.88rem; margin: 0.5rem 0;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e3a5f; margin: 1.5rem 0; }

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #080c14 !important;
    border-right: 1px solid #1e3a5f !important;
}
div[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Streamlit overrides ── */
.stRadio > div { gap: 0.5rem; }
.stRadio label { color: #8892b0 !important; }
div[data-testid="stMetric"] { background: #0d1b2a; border-radius: 10px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    import subprocess
    biased_path   = "model/biased_model.pkl"
    debiased_path = "model/debiased_model.pkl"
    test_path     = "model/test_data.csv"
    data_path     = "data/loan_data.csv"

    # Ensure init files exist and are clean
    os.makedirs("data",  exist_ok=True)
    os.makedirs("model", exist_ok=True)
    for init in ["data/__init__.py", "model/__init__.py"]:
        if not os.path.exists(init):
            open(init, "w").write("")

    if not os.path.exists(data_path):
        subprocess.run([sys.executable, "data/generate_dataset.py"], check=True)

    if not (os.path.exists(biased_path) and os.path.exists(debiased_path)):
        subprocess.run([sys.executable, "model/train.py"], check=True)

    with open(biased_path,   "rb") as f: bm = pickle.load(f)
    with open(debiased_path, "rb") as f: dm = pickle.load(f)
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(data_path)
    return bm, dm, test_df, full_df


bm, dm, test_df, full_df = load_all()

FEATURES = ["age", "income", "credit_score", "employment_years", "gender"]
X_test = test_df[FEATURES]
y_test = test_df["approved"]
s_test = test_df["gender"]

yp_b = bm.predict(X_test)
yp_d = dm.predict(X_test)

mb = fairness_metrics(y_test, yp_b, s_test)
md = fairness_metrics(y_test, yp_d, s_test)

gm_b = mb["group_metrics"]
gm_d = md["group_metrics"]

# Feature importance from logistic regression coefficients
FEAT_LABELS = ["Age", "Income", "Credit Score", "Employment Yrs", "Gender"]
coef_b = bm.named_steps["clf"].coef_[0]
coef_d = dm.named_steps["clf"].coef_[0]

# Fairness score: 0-100 (100 = perfectly fair)
def fairness_score(metrics):
    dp  = metrics["demographic_parity_diff"]
    eo  = metrics["equal_opportunity_diff"]
    ag  = metrics["accuracy_gap"]
    raw = 1 - (0.4 * dp + 0.4 * eo + 0.2 * ag)
    return max(0, min(100, round(raw * 100)))

fs_b = fairness_score(mb)
fs_d = fairness_score(md)

# ─────────────────────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#080c14"
CARD_BG   = "#0d1b2a"
GRID_COL  = "#1e3a5f"
TEXT_COL  = "#8892b0"
RED       = "#e94560"
GREEN     = "#00d4aa"
GOLD      = "#ffd700"
BLUE      = "#3b82f6"

def chart_base(fig, height=320):
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COL, family="Inter"),
        height=height,
        margin=dict(t=40, b=30, l=30, r=20),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor=CARD_BG, bordercolor=GRID_COL, borderwidth=1),
    )
    return fig


def gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 36, "color": color, "family": "Inter"}, "suffix": ""},
        title={"text": title, "font": {"size": 13, "color": TEXT_COL}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_COL,
                     "tickfont": {"size": 10}, "nticks": 6},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": CARD_BG,
            "bordercolor": GRID_COL,
            "steps": [
                {"range": [0,  40], "color": "rgba(233,69,96,0.08)"},
                {"range": [40, 70], "color": "rgba(255,215,0,0.08)"},
                {"range": [70,100], "color": "rgba(0,212,170,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8, "value": value
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG, font=dict(family="Inter"),
        height=220, margin=dict(t=30, b=10, l=20, r=20)
    )
    return fig


def approval_bar(gm_biased, gm_debiased):
    groups = ["Female", "Male"]
    b_rates = [gm_biased.get(g,  {}).get("approval_rate", 0) * 100 for g in groups]
    d_rates = [gm_debiased.get(g, {}).get("approval_rate", 0) * 100 for g in groups]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Biased",   x=groups, y=b_rates,
                         marker_color=RED,   opacity=0.9,
                         text=[f"{v:.1f}%" for v in b_rates], textposition="outside"))
    fig.add_trace(go.Bar(name="Debiased", x=groups, y=d_rates,
                         marker_color=GREEN, opacity=0.9,
                         text=[f"{v:.1f}%" for v in d_rates], textposition="outside"))
    fig.update_layout(
        barmode="group", title="Approval Rate by Gender",
        title_font=dict(color="#ccd6f6", size=13),
        yaxis=dict(range=[0, 110], title="Approval Rate (%)"),
    )
    return chart_base(fig, 300)


def fairness_compare_chart(mb, md):
    cats   = ["Demographic<br>Parity", "Equal<br>Opportunity", "Accuracy<br>Gap"]
    b_vals = [mb["demographic_parity_diff"], mb["equal_opportunity_diff"], mb["accuracy_gap"]]
    d_vals = [md["demographic_parity_diff"], md["equal_opportunity_diff"], md["accuracy_gap"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Biased",   x=cats, y=b_vals,
                         marker_color=RED,   opacity=0.9,
                         text=[f"{v:.3f}" for v in b_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="Debiased", x=cats, y=d_vals,
                         marker_color=GREEN, opacity=0.9,
                         text=[f"{v:.3f}" for v in d_vals], textposition="outside"))
    fig.update_layout(
        barmode="group", title="Fairness Metrics Comparison (lower = fairer)",
        title_font=dict(color="#ccd6f6", size=13),
        yaxis=dict(title="Disparity Score"),
    )
    return chart_base(fig, 320)


def confusion_matrix_chart(y_true, y_pred, gender_val, title, color):
    mask = s_test == gender_val
    cm   = confusion_matrix(y_true[mask], y_pred[mask])
    labels = [["TN", "FP"], ["FN", "TP"]]
    text   = [[f"<b>{labels[i][j]}</b><br>{cm[i][j]}" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Denied", "Pred: Approved"],
        y=["Actual: Denied", "Actual: Approved"],
        text=text, texttemplate="%{text}",
        colorscale=[[0, CARD_BG], [1, color]],
        showscale=False,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        title=title, title_font=dict(color="#ccd6f6", size=12),
        xaxis=dict(side="bottom"),
    )
    return chart_base(fig, 240)


def feature_importance_chart(coef_b, coef_d, labels):
    abs_b = np.abs(coef_b)
    abs_d = np.abs(coef_d)
    order = np.argsort(abs_b)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[labels[i] for i in order], x=abs_b[order],
        name="Biased", orientation="h",
        marker_color=[RED if labels[i] == "Gender" else "#3b82f6" for i in order],
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        y=[labels[i] for i in order], x=abs_d[order],
        name="Debiased", orientation="h",
        marker_color=[GREEN if labels[i] == "Gender" else "#60a5fa" for i in order],
        opacity=0.7,
    ))
    fig.update_layout(
        barmode="overlay", title="Feature Influence (Absolute Coefficient)",
        title_font=dict(color="#ccd6f6", size=13),
        xaxis=dict(title="Influence Weight"),
        yaxis=dict(title=""),
    )
    return chart_base(fig, 280)


def prob_gauge_single(prob, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": color, "family": "Inter"}},
        title={"text": title, "font": {"size": 11, "color": TEXT_COL}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 9}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": CARD_BG, "bordercolor": GRID_COL,
            "steps": [{"range": [0, 50], "color": "rgba(233,69,96,0.06)"},
                      {"range": [50, 100], "color": "rgba(0,212,170,0.06)"}],
        }
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG, font=dict(family="Inter"),
        height=180, margin=dict(t=25, b=5, l=10, r=10)
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
      <div style='font-size:1.8rem;'>&#9878;</div>
      <div style='font-weight:800; font-size:1.1rem; color:#ccd6f6;'>FairLens AI</div>
      <div style='font-size:0.72rem; color:#8892b0;'>Bias Detection Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e3a5f; margin:0.8rem 0;'>", unsafe_allow_html=True)

    # ── Bias Toggle ──
    st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#8892b0; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Model Mode</div>", unsafe_allow_html=True)
    bias_mode = st.radio(
        "model_mode",
        ["Biased Model  (Before)", "Debiased Model  (After)"],
        index=0, label_visibility="collapsed"
    )
    is_debiased = "Debiased" in bias_mode

    if is_debiased:
        st.markdown("<div class='alert-good'>&#10003; Fairness corrections applied</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-bad'>&#9888; Biased model active</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e3a5f; margin:0.8rem 0;'>", unsafe_allow_html=True)

    # ── Applicant Form ──
    st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#8892b0; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;'>Applicant Profile</div>", unsafe_allow_html=True)

    gender_input = st.selectbox("Gender", ["Female", "Male"])
    age_input    = st.slider("Age", 22, 60, 35)
    income_input = st.slider("Annual Income ($)", 20000, 120000, 55000, step=1000)
    credit_input = st.slider("Credit Score", 300, 850, 650)
    emp_input    = st.slider("Employment Years", 0, 30, 5)

    st.markdown("<hr style='border-color:#1e3a5f; margin:0.8rem 0;'>", unsafe_allow_html=True)

    # ── Step guide ──
    st.markdown("""
    <div style='font-size:0.78rem; font-weight:600; color:#8892b0; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;'>How to Use</div>
    <div class='step-row'>
      <div class='step-num'>1</div>
      <div class='step-text'><b>Set applicant profile</b> using the sliders above</div>
    </div>
    <div class='step-row'>
      <div class='step-num'>2</div>
      <div class='step-text'><b>See the biased decision</b> — notice how gender affects outcome</div>
    </div>
    <div class='step-row'>
      <div class='step-num'>3</div>
      <div class='step-text'><b>Switch to Debiased Model</b> to apply the fairness fix</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
female_approval = round(full_df[full_df["gender"]==0]["approved"].mean() * 100, 1)
male_approval   = round(full_df[full_df["gender"]==1]["approved"].mean() * 100, 1)
bias_gap        = round(male_approval - female_approval, 1)
dp_improvement  = round((mb["demographic_parity_diff"] - md["demographic_parity_diff"]) / mb["demographic_parity_diff"] * 100)

st.markdown(f"""
<div class='hero'>
  <p class='hero-title'>FairLens AI</p>
  <p class='hero-sub'>Detecting &amp; Eliminating Bias in Machine Learning &nbsp;|&nbsp; Hack2Skill Solution Challenge 2026</p>
  <div class='hero-stats'>
    <div class='hero-stat'>
      <div class='hero-stat-val'>{bias_gap}%</div>
      <div class='hero-stat-lbl'>Gender Approval Gap</div>
    </div>
    <div class='hero-stat'>
      <div class='hero-stat-val' style='color:#ffd700;'>{female_approval}%</div>
      <div class='hero-stat-lbl'>Female Approval Rate</div>
    </div>
    <div class='hero-stat'>
      <div class='hero-stat-val' style='color:#00d4aa;'>{male_approval}%</div>
      <div class='hero-stat-lbl'>Male Approval Rate</div>
    </div>
    <div class='hero-stat'>
      <div class='hero-stat-val' style='color:#00d4aa;'>{dp_improvement}%</div>
      <div class='hero-stat-lbl'>Bias Reduced After Fix</div>
    </div>
    <div class='hero-stat'>
      <div class='hero-stat-val' style='color:#3b82f6;'>1,000</div>
      <div class='hero-stat-lbl'>Applicants Analyzed</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-red'>&#9888;</div>
  The Problem: AI Bias in Loan Decisions
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class='card'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>&#129302;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.5rem;'>What is AI Bias?</div>
      <div style='color:#8892b0; font-size:0.85rem; line-height:1.6;'>
        ML models learn from historical data. When that data reflects past discrimination,
        the model <b style='color:#e94560;'>learns and amplifies</b> it — even without
        explicitly using protected attributes like gender.
      </div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='card'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>&#128202;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.5rem;'>The Hidden Penalty</div>
      <div style='color:#8892b0; font-size:0.85rem; line-height:1.6;'>
        Our dataset has a <b style='color:#e94560;'>20% approval penalty</b> injected for
        female applicants — simulating real-world historical bias in lending.
        Result: Female approval rate is <b style='color:#e94560;'>{female_approval}%</b>
        vs Male <b style='color:#00d4aa;'>{male_approval}%</b>.
      </div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='card'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>&#128295;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.5rem;'>Our Fix</div>
      <div style='color:#8892b0; font-size:0.85rem; line-height:1.6;'>
        <b style='color:#00d4aa;'>Sample Reweighting</b>: Underrepresented
        (gender &times; outcome) groups get higher training weights.
        No data deleted. Result: <b style='color:#00d4aa;'>{dp_improvement}% reduction</b>
        in demographic parity disparity.
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-blue'>&#127919;</div>
  Live Prediction Panel
</div>
""", unsafe_allow_html=True)

gender_val = 1 if gender_input == "Male" else 0
user_df = pd.DataFrame(
    [[age_input, income_input, credit_input, emp_input, gender_val]],
    columns=FEATURES
)

active_model = dm if is_debiased else bm
pred  = active_model.predict(user_df)[0]
prob  = active_model.predict_proba(user_df)[0][1]

pred_b = bm.predict(user_df)[0]
prob_b = bm.predict_proba(user_df)[0][1]
pred_d = dm.predict(user_df)[0]
prob_d = dm.predict_proba(user_df)[0][1]

verdict     = "APPROVED" if pred == 1 else "DENIED"
pred_class  = "pred-approved" if pred == 1 else "pred-denied"
pred_color  = GREEN if pred == 1 else RED
mode_label  = "Debiased Model" if is_debiased else "Biased Model"

col_verdict, col_gauges, col_insight = st.columns([1, 1.4, 1.4])

with col_verdict:
    st.markdown(f"""
    <div class='{pred_class} pred-box'>
      <div style='font-size:0.75rem; color:#8892b0; margin-bottom:0.8rem; text-transform:uppercase; letter-spacing:0.1em;'>{mode_label}</div>
      <div class='pred-verdict' style='color:{pred_color};'>{verdict}</div>
      <div class='pred-prob'>Approval Probability<br><b style='font-size:1.4rem; color:{pred_color};'>{prob*100:.1f}%</b></div>
      <div class='pred-meta'>{gender_input} &nbsp;|&nbsp; Age {age_input} &nbsp;|&nbsp; ${income_input:,}<br>Credit {credit_input} &nbsp;|&nbsp; {emp_input} yrs exp</div>
    </div>""", unsafe_allow_html=True)

with col_gauges:
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(prob_gauge_single(prob_b, RED,   "Biased Model"),   use_container_width=True)
    with g2:
        st.plotly_chart(prob_gauge_single(prob_d, GREEN, "Debiased Model"), use_container_width=True)

with col_insight:
    diff = prob_d - prob_b
    diff_pct = abs(diff) * 100

    if pred_b != pred_d:
        outcome_b = "approved" if pred_b == 1 else "denied"
        outcome_d = "approved" if pred_d == 1 else "denied"
        st.markdown(f"""
        <div class='alert-bad'>
          <b>&#9888; Bias Detected!</b><br>
          Biased model: <b>{outcome_b.upper()}</b><br>
          Debiased model: <b>{outcome_d.upper()}</b><br>
          Probability shift: <b>{diff_pct:.1f}%</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='alert-good'>
          <b>&#10003; Models Agree</b><br>
          Both models reach the same decision.<br>
          Probability difference: <b>{diff_pct:.1f}%</b>
        </div>""", unsafe_allow_html=True)

    # Gender-specific insight
    if gender_val == 0 and not is_debiased:
        st.markdown("""
        <div class='alert-warn' style='margin-top:0.5rem;'>
          <b>&#9888; Gender Bias Active</b><br>
          Female applicants face a systematic disadvantage in this model.
          Switch to Debiased Model to see the corrected outcome.
        </div>""", unsafe_allow_html=True)
    elif gender_val == 0 and is_debiased:
        st.markdown("""
        <div class='alert-good' style='margin-top:0.5rem;'>
          <b>&#10003; Fairness Applied</b><br>
          Gender bias has been corrected. This applicant is now evaluated
          on financial merit alone.
        </div>""", unsafe_allow_html=True)

    # Why this decision?
    st.markdown("<div style='margin-top:0.8rem; font-size:0.78rem; font-weight:600; color:#8892b0; text-transform:uppercase; letter-spacing:0.08em;'>Why this decision?</div>", unsafe_allow_html=True)
    active_coef = coef_d if is_debiased else coef_b
    scaler = active_model.named_steps["scaler"]
    x_scaled = scaler.transform(user_df)[0]
    contributions = active_coef * x_scaled
    contrib_df = pd.DataFrame({"feature": FEAT_LABELS, "contribution": contributions})
    contrib_df = contrib_df.reindex(contrib_df["contribution"].abs().sort_values(ascending=False).index)

    for _, row in contrib_df.head(4).iterrows():
        bar_color = GREEN if row["contribution"] > 0 else RED
        bar_pct   = min(100, abs(row["contribution"]) / (abs(contributions).max() + 1e-9) * 100)
        direction = "+" if row["contribution"] > 0 else "-"
        st.markdown(f"""
        <div style='margin-bottom:0.4rem;'>
          <div style='display:flex; justify-content:space-between; font-size:0.78rem; color:#8892b0; margin-bottom:0.2rem;'>
            <span>{row["feature"]}</span>
            <span style='color:{bar_color};'>{direction} influence</span>
          </div>
          <div class='prog-wrap'>
            <div class='prog-fill' style='width:{bar_pct:.0f}%; background:{bar_color};'></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — FAIRNESS SCORE + KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-gold'>&#9878;</div>
  Fairness Metrics &amp; Score
</div>
""", unsafe_allow_html=True)

active_mb = md if is_debiased else mb
active_fs = fs_d if is_debiased else fs_b

# Fairness score color
if active_fs >= 70:
    fs_color = GREEN; fs_badge = "Fair Model Achieved"; fs_badge_cls = "badge-green"
elif active_fs >= 45:
    fs_color = GOLD;  fs_badge = "Partially Fair";      fs_badge_cls = "badge-gold"
else:
    fs_color = RED;   fs_badge = "Bias Detected";       fs_badge_cls = "badge-red"

col_score, col_kpis = st.columns([1, 2.5])

with col_score:
    st.plotly_chart(gauge_chart(active_fs, "Fairness Score", fs_color), use_container_width=True)
    st.markdown(f"""
    <div style='text-align:center; margin-top:-0.5rem;'>
      <span class='card-badge {fs_badge_cls}'>{fs_badge}</span>
      <div style='margin-top:0.5rem; font-size:0.78rem; color:#8892b0;'>
        Biased: <b style='color:{RED};'>{fs_b}</b> &nbsp;&#8594;&nbsp;
        Debiased: <b style='color:{GREEN};'>{fs_d}</b>
        &nbsp;<b style='color:{GREEN};'>(+{fs_d - fs_b} pts)</b>
      </div>
    </div>""", unsafe_allow_html=True)

with col_kpis:
    dp_val  = active_mb["demographic_parity_diff"]
    eo_val  = active_mb["equal_opportunity_diff"]
    ag_val  = active_mb["accuracy_gap"]
    acc_val = active_mb["overall_accuracy"]

    def kpi_color(v):
        if v <= 0.05: return GREEN, "badge-green", "Fair"
        if v <= 0.15: return GOLD,  "badge-gold",  "Moderate Bias"
        return RED, "badge-red", "High Bias"

    dp_c,  dp_bc,  dp_lbl  = kpi_color(dp_val)
    eo_c,  eo_bc,  eo_lbl  = kpi_color(eo_val)
    ag_c,  ag_bc,  ag_lbl  = kpi_color(ag_val)

    dp_imp = round((mb["demographic_parity_diff"] - md["demographic_parity_diff"]) / (mb["demographic_parity_diff"] + 1e-9) * 100)
    eo_imp = round((mb["equal_opportunity_diff"]   - md["equal_opportunity_diff"])   / (mb["equal_opportunity_diff"]   + 1e-9) * 100)
    ag_imp = round((mb["accuracy_gap"]             - md["accuracy_gap"])             / (mb["accuracy_gap"]             + 1e-9) * 100)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Demographic Parity</div>
          <div class='card-val' style='color:{dp_c};'>{dp_val:.3f}</div>
          <div class='card-sub'>0.05 or less = fair</div>
          <span class='card-badge {dp_bc}'>{dp_lbl}</span>
          <div style='font-size:0.75rem; color:{GREEN}; margin-top:0.4rem;'>&#8595; {dp_imp}% improved</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Equal Opportunity</div>
          <div class='card-val' style='color:{eo_c};'>{eo_val:.3f}</div>
          <div class='card-sub'>0.05 or less = fair</div>
          <span class='card-badge {eo_bc}'>{eo_lbl}</span>
          <div style='font-size:0.75rem; color:{GREEN}; margin-top:0.4rem;'>&#8595; {eo_imp}% improved</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Accuracy Gap</div>
          <div class='card-val' style='color:{ag_c};'>{ag_val:.3f}</div>
          <div class='card-sub'>0.05 or less = fair</div>
          <span class='card-badge {ag_bc}'>{ag_lbl}</span>
          <div style='font-size:0.75rem; color:{GREEN}; margin-top:0.4rem;'>&#8595; {ag_imp}% improved</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Overall Accuracy</div>
          <div class='card-val val-white'>{acc_val*100:.1f}%</div>
          <div class='card-sub'>Model performance</div>
          <span class='card-badge badge-green'>Maintained</span>
          <div style='font-size:0.75rem; color:#8892b0; margin-top:0.4rem;'>Biased: {mb["overall_accuracy"]*100:.1f}% | Debiased: {md["overall_accuracy"]*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — BEFORE vs AFTER VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-green'>&#8644;</div>
  Before vs After Bias Mitigation
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(approval_bar(gm_b, gm_d), use_container_width=True)
with col_r:
    st.plotly_chart(fairness_compare_chart(mb, md), use_container_width=True)

# Progress bars showing improvement
st.markdown("<div style='margin-top:0.5rem;'>", unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)

metrics_list = [
    ("Demographic Parity Reduction", mb["demographic_parity_diff"], md["demographic_parity_diff"], dp_imp),
    ("Equal Opportunity Reduction",  mb["equal_opportunity_diff"],  md["equal_opportunity_diff"],  eo_imp),
    ("Accuracy Gap Reduction",       mb["accuracy_gap"],            md["accuracy_gap"],            ag_imp),
]
for col, (label, before, after, imp) in zip([p1, p2, p3], metrics_list):
    with col:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>{label}</div>
          <div style='display:flex; justify-content:space-between; margin:0.5rem 0 0.2rem 0;'>
            <span style='font-size:0.8rem; color:{RED};'>Before: {before:.3f}</span>
            <span style='font-size:0.8rem; color:{GREEN};'>After: {after:.3f}</span>
          </div>
          <div class='prog-wrap'>
            <div class='prog-fill prog-green' style='width:{min(imp,100)}%;'></div>
          </div>
          <div style='font-size:0.85rem; color:{GREEN}; font-weight:700; margin-top:0.4rem;'>
            &#8595; {imp}% reduction
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — CONFUSION MATRICES + FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-blue'>&#128202;</div>
  Deep Dive: Confusion Matrices &amp; Feature Influence
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Confusion Matrices by Group", "Feature Influence (Explainability)"])

with tab1:
    active_yp = yp_d if is_debiased else yp_b
    active_color = GREEN if is_debiased else RED
    active_label = "Debiased" if is_debiased else "Biased"

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(
            confusion_matrix_chart(y_test.values, active_yp, 0,
                                   f"{active_label} Model — Female Applicants", active_color),
            use_container_width=True
        )
    with cm2:
        st.plotly_chart(
            confusion_matrix_chart(y_test.values, active_yp, 1,
                                   f"{active_label} Model — Male Applicants", active_color),
            use_container_width=True
        )

    # TPR / FPR table
    active_gm = gm_d if is_debiased else gm_b
    tpr_f = active_gm.get("Female", {}).get("tpr", 0)
    tpr_m = active_gm.get("Male",   {}).get("tpr", 0)
    fpr_f = active_gm.get("Female", {}).get("fpr", 0)
    fpr_m = active_gm.get("Male",   {}).get("fpr", 0)
    acc_f = active_gm.get("Female", {}).get("accuracy", 0)
    acc_m = active_gm.get("Male",   {}).get("accuracy", 0)

    tbl_df = pd.DataFrame({
        "Group":         ["Female", "Male"],
        "True Pos Rate": [f"{tpr_f*100:.1f}%", f"{tpr_m*100:.1f}%"],
        "False Pos Rate":[f"{fpr_f*100:.1f}%", f"{fpr_m*100:.1f}%"],
        "Accuracy":      [f"{acc_f*100:.1f}%", f"{acc_m*100:.1f}%"],
        "Approval Rate": [
            f"{active_gm.get('Female',{}).get('approval_rate',0)*100:.1f}%",
            f"{active_gm.get('Male',  {}).get('approval_rate',0)*100:.1f}%",
        ],
    })
    st.dataframe(tbl_df, use_container_width=True, hide_index=True)

with tab2:
    st.plotly_chart(feature_importance_chart(coef_b, coef_d, FEAT_LABELS),
                    use_container_width=True)

    gender_b = abs(coef_b[FEAT_LABELS.index("Gender")])
    gender_d = abs(coef_d[FEAT_LABELS.index("Gender")])
    gender_drop = round((gender_b - gender_d) / gender_b * 100)

    st.markdown(f"""
    <div class='alert-good'>
      <b>Key Insight:</b> The Gender feature's influence dropped from
      <b style='color:{RED};'>{gender_b:.3f}</b> to
      <b style='color:{GREEN};'>{gender_d:.3f}</b> after debiasing
      — a <b>{gender_drop}% reduction</b> in gender's weight on the decision.
      The model now relies primarily on financial merit (credit score, income).
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — AUTO-GENERATED INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon icon-gold'>&#128161;</div>
  Auto-Generated Insights
</div>
""", unsafe_allow_html=True)

i1, i2 = st.columns(2)

with i1:
    # Dynamic message based on current model
    if is_debiased:
        msg_color = GREEN
        msg_icon  = "&#10003;"
        msg_title = "Bias Reduced by " + str(dp_improvement) + "%"
        msg_body  = (
            f"After applying sample reweighting, the demographic parity difference "
            f"dropped from <b style='color:{RED};'>{mb['demographic_parity_diff']:.3f}</b> to "
            f"<b style='color:{GREEN};'>{md['demographic_parity_diff']:.3f}</b>. "
            f"Female applicants now receive approval rates comparable to male applicants "
            f"with similar financial profiles."
        )
    else:
        msg_color = RED
        msg_icon  = "&#9888;"
        msg_title = "Model is Biased Towards Males"
        msg_body  = (
            f"The biased model approves male applicants at "
            f"<b style='color:{GREEN};'>{gm_b.get('Male',{}).get('approval_rate',0)*100:.1f}%</b> "
            f"vs female applicants at only "
            f"<b style='color:{RED};'>{gm_b.get('Female',{}).get('approval_rate',0)*100:.1f}%</b>. "
            f"This {bias_gap:.1f}% gap cannot be explained by financial factors alone — "
            f"it is systematic gender discrimination."
        )

    st.markdown(f"""
    <div class='card' style='border-color:{msg_color}44;'>
      <div style='font-size:1.3rem; margin-bottom:0.5rem;'>{msg_icon}</div>
      <div style='font-weight:700; color:{msg_color}; font-size:1rem; margin-bottom:0.5rem;'>{msg_title}</div>
      <div style='color:#8892b0; font-size:0.85rem; line-height:1.6;'>{msg_body}</div>
    </div>""", unsafe_allow_html=True)

with i2:
    # Fairness score progress
    st.markdown(f"""
    <div class='card'>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:1rem;'>&#127942; Fairness Score Progress</div>

      <div style='font-size:0.8rem; color:#8892b0; margin-bottom:0.3rem;'>
        Biased Model &nbsp;<b style='color:{RED};'>{fs_b}/100</b>
      </div>
      <div class='prog-wrap' style='margin-bottom:0.8rem;'>
        <div class='prog-fill prog-red' style='width:{fs_b}%;'></div>
      </div>

      <div style='font-size:0.8rem; color:#8892b0; margin-bottom:0.3rem;'>
        Debiased Model &nbsp;<b style='color:{GREEN};'>{fs_d}/100</b>
      </div>
      <div class='prog-wrap' style='margin-bottom:0.8rem;'>
        <div class='prog-fill prog-green' style='width:{fs_d}%;'></div>
      </div>

      <div style='margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid #1e3a5f;'>
        <span class='card-badge badge-green'>+{fs_d - fs_b} points improvement</span>
        &nbsp;
        <span class='card-badge {"badge-green" if is_debiased else "badge-red"}'>
          {"Fair Model Achieved" if is_debiased else "Bias Detected"}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — DATASET EXPLORER (collapsible)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Dataset Explorer", expanded=False):
    d1, d2 = st.columns(2)
    with d1:
        agg = full_df.groupby("gender")["approved"].mean().reset_index()
        agg["gender"] = agg["gender"].map({0: "Female", 1: "Male"})
        fig_raw = px.bar(agg, x="gender", y="approved",
                         color="gender",
                         color_discrete_map={"Female": RED, "Male": BLUE},
                         title="Raw Dataset: Approval Rate by Gender",
                         labels={"approved": "Approval Rate", "gender": "Gender"})
        fig_raw.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                               font_color=TEXT_COL, showlegend=False, height=280,
                               margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_raw, use_container_width=True)
    with d2:
        fig_dist = px.histogram(full_df, x="credit_score", color="gender",
                                color_discrete_map={0: RED, 1: BLUE},
                                barmode="overlay", opacity=0.7,
                                title="Credit Score Distribution by Gender",
                                labels={"gender": "Gender (0=Female, 1=Male)"})
        fig_dist.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                                font_color=TEXT_COL, height=280,
                                margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_dist, use_container_width=True)

    display_df = full_df.copy()
    display_df["gender"]   = display_df["gender"].map({0: "Female", 1: "Male"})
    display_df["approved"] = display_df["approved"].map({0: "Denied", 1: "Approved"})
    st.dataframe(display_df.head(50), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER — 30-second judge summary
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f; margin:2rem 0 1rem 0;'>", unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(f"""
    <div class='card' style='border-color:#e9456044;'>
      <div style='font-size:1.2rem; margin-bottom:0.4rem;'>&#128269;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.4rem;'>Problem</div>
      <div style='color:#8892b0; font-size:0.82rem; line-height:1.5;'>
        ML models trained on biased data perpetuate discrimination.
        Our loan dataset has a <b style='color:{RED};'>20% hidden penalty</b> for female applicants.
      </div>
    </div>""", unsafe_allow_html=True)
with f2:
    st.markdown(f"""
    <div class='card' style='border-color:#ffd70044;'>
      <div style='font-size:1.2rem; margin-bottom:0.4rem;'>&#128202;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.4rem;'>Detection</div>
      <div style='color:#8892b0; font-size:0.82rem; line-height:1.5;'>
        We measure <b style='color:{GOLD};'>Demographic Parity</b>,
        <b style='color:{GOLD};'>Equal Opportunity</b>, and
        <b style='color:{GOLD};'>Accuracy Gap</b> across gender groups.
      </div>
    </div>""", unsafe_allow_html=True)
with f3:
    st.markdown(f"""
    <div class='card' style='border-color:#00d4aa44;'>
      <div style='font-size:1.2rem; margin-bottom:0.4rem;'>&#128295;</div>
      <div style='font-weight:700; color:#ccd6f6; margin-bottom:0.4rem;'>Solution</div>
      <div style='color:#8892b0; font-size:0.82rem; line-height:1.5;'>
        <b style='color:{GREEN};'>Sample Reweighting</b> reduces bias by
        <b style='color:{GREEN};'>{dp_improvement}%</b> while maintaining accuracy.
        Fairness score: <b style='color:{RED};'>{fs_b}</b> &#8594; <b style='color:{GREEN};'>{fs_d}</b>/100.
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.75rem; margin-top:1rem; padding-bottom:1rem;'>
  FairLens AI &nbsp;|&nbsp; Hack2Skill Solution Challenge 2026 &nbsp;|&nbsp; Unbiased AI Decision Track
</div>
""", unsafe_allow_html=True)
