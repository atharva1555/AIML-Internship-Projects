import streamlit as st
import numpy as np
import pickle
import os
import time

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #f0f4ff;
    --white:      #ffffff;
    --border:     #dde3f0;
    --blue:       #3b5bdb;
    --blue-dark:  #2f4ac5;
    --blue-light: #eef2ff;
    --teal:       #0891b2;
    --text:       #111827;
    --muted:      #6b7280;
    --green:      #059669;
    --green-bg:   #ecfdf5;
    --orange:     #ea580c;
    --red:        #dc2626;
    --yellow:     #d97706;
    --shadow:     0 4px 24px rgba(59,91,219,0.10);
    --shadow-lg:  0 12px 48px rgba(59,91,219,0.18);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1380px; }

/* ── Animated Background Grid ── */
.bg-grid {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(59,91,219,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,91,219,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: var(--shadow);
    animation: slideDown 0.5s ease;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-logo {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--blue), var(--teal));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 12px rgba(59,91,219,0.35);
}
.nav-brand { font-size: 1.15rem; font-weight: 800; color: var(--text); letter-spacing: -0.3px; }
.nav-brand span { color: var(--blue); }
.nav-tagline { font-size: 0.72rem; color: var(--muted); font-weight: 400; margin-top: 1px; }
.nav-right { display: flex; align-items: center; gap: 10px; }
.nav-pill {
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.nav-pill.blue { background: var(--blue-light); color: var(--blue); border: 1px solid #c5d0ff; }
.nav-pill.green { background: var(--green-bg); color: var(--green); border: 1px solid #a7f3d0; }
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

/* ── Hero ── */
.hero {
    background: linear-gradient(130deg, #1e3a8a 0%, #2563eb 45%, #0891b2 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.6s ease 0.1s both;
    box-shadow: var(--shadow-lg);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero::before {
    content: '';
    position: absolute;
    width: 380px; height: 380px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
    top: -120px; right: -80px;
}
.hero::after {
    content: '';
    position: absolute;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
    bottom: -80px; right: 220px;
}
.hero-inner { position: relative; z-index: 1; }
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 5px 16px;
    border-radius: 100px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.95);
    margin-bottom: 1.3rem;
    width: fit-content;
}
.hero h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: white !important;
    line-height: 1.05 !important;
    margin: 0 0 1rem 0 !important;
    letter-spacing: -1px !important;
}
.hero h1 span {
    background: linear-gradient(90deg, #93c5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: 1rem;
    color: rgba(255,255,255,0.72);
    max-width: 500px;
    line-height: 1.75;
    margin: 0 0 2rem 0;
}
.hero-stats { display: flex; gap: 2rem; }
.hstat-val { font-size: 1.6rem; font-weight: 800; color: white; line-height: 1; }
.hstat-label { font-size: 0.72rem; color: rgba(255,255,255,0.55); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
.hstat-div { width: 1px; background: rgba(255,255,255,0.15); }

/* ── How it Works Bar ── */
.how-bar {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.8rem;
    animation: fadeUp 0.6s ease 0.2s both;
}
.how-step {
    flex: 1;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.how-step:hover { box-shadow: var(--shadow); transform: translateY(-2px); }
.how-num {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--blue), var(--teal));
    color: white;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.how-text { font-size: 0.8rem; font-weight: 500; color: var(--muted); }

/* ── Form Cards ── */
.fcard {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    animation: fadeUp 0.6s ease 0.3s both;
    transition: box-shadow 0.2s;
}
.fcard:hover { box-shadow: var(--shadow); }
.fcard-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.fcard-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.fcard-icon.blue  { background: var(--blue-light); }
.fcard-icon.teal  { background: #ecfeff; }
.fcard-title { font-size: 1rem; font-weight: 700; color: var(--text); letter-spacing: -0.2px; }
.fcard-sub { font-size: 0.76rem; color: var(--muted); margin-top: 2px; }

/* ── Widget Overrides ── */
.stSelectbox > label,
.stNumberInput > label,
.stSlider > label {
    font-size: 0.73rem !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 5px !important;
}
.stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 11px !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
    background: white !important;
}
.stNumberInput input {
    background: var(--bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 11px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
    background: white !important;
}
.stSlider > div > div > div { background: #c7d2fe !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, var(--blue), var(--teal)) !important; }

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue) 0%, var(--teal) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    padding: 0.9rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 20px rgba(59,91,219,0.35) !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 32px rgba(59,91,219,0.45) !important;
}

/* ── Result Card ── */
.res-card {
    border-radius: 22px;
    padding: 2.4rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-lg);
    animation: fadeUp 0.5s ease;
}
.res-card.high   { background: linear-gradient(135deg, #064e3b, #059669); }
.res-card.medium { background: linear-gradient(135deg, #78350f, #d97706); }
.res-card.low    { background: linear-gradient(135deg, #7f1d1d, #dc2626); }
.res-card::before {
    content: '';
    position: absolute;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
    top: -80px; right: -60px;
}
.res-eyebrow { font-size: 0.7rem; letter-spacing: 2.5px; text-transform: uppercase; color: rgba(255,255,255,0.6); margin-bottom: 0.7rem; font-weight: 600; }
.res-amount { font-size: 4rem; font-weight: 900; line-height: 1; margin-bottom: 0.4rem; letter-spacing: -2px; }
.res-currency { font-size: 2rem; font-weight: 500; vertical-align: super; margin-right: 3px; letter-spacing: 0; }
.res-sub { font-size: 0.82rem; color: rgba(255,255,255,0.55); margin-bottom: 1.5rem; }
.res-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 5px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Metric Pills ── */
.metrics-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.5s ease 0.1s both;
}
.metric-pill {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-pill:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
.mp-icon { font-size: 1.3rem; margin-bottom: 4px; }
.mp-val { font-size: 0.95rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.mp-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-top: 3px; font-weight: 600; }

/* ── Sales Meter ── */
.meter-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    animation: fadeUp 0.5s ease 0.2s both;
}
.meter-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; }
.meter-title { font-size: 0.78rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.meter-pct { font-size: 0.82rem; font-weight: 700; color: var(--blue); font-family: 'JetBrains Mono', monospace; }
.meter-track { width: 100%; height: 10px; background: var(--bg); border-radius: 100px; overflow: hidden; border: 1px solid var(--border); }
.meter-fill { height: 100%; border-radius: 100px; }
.meter-labels { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.65rem; color: var(--muted); font-weight: 600; }

/* ── Summary ── */
.sum-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    animation: fadeUp 0.5s ease 0.3s both;
}
.sum-title { font-size: 0.85rem; font-weight: 800; color: var(--text); margin-bottom: 0.9rem; display: flex; align-items: center; gap: 6px; }
.sum-row { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 0.84rem; }
.sum-row:last-child { border-bottom: none; }
.sum-key { color: var(--muted); font-weight: 500; }
.sum-val { color: var(--text); font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: var(--bg); padding: 2px 9px; border-radius: 6px; border: 1px solid var(--border); }

/* ── Recommendation ── */
.reco-card {
    background: linear-gradient(135deg, var(--blue-light), #f0fdff);
    border: 1px solid #c5d0ff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    animation: fadeUp 0.5s ease 0.4s both;
}
.reco-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.reco-text { font-size: 0.82rem; color: #1e3a8a; line-height: 1.6; font-weight: 500; }
.reco-text strong { font-weight: 800; }

/* ── Empty State ── */
.empty-wrap {
    background: var(--white);
    border: 2px dashed var(--border);
    border-radius: 22px;
    padding: 4rem 2rem;
    text-align: center;
    animation: fadeUp 0.5s ease 0.4s both;
}
.empty-emoji { font-size: 3.5rem; margin-bottom: 1.2rem; display: block; }
.empty-h { font-size: 1.15rem; font-weight: 800; color: var(--text); margin-bottom: 0.5rem; letter-spacing: -0.3px; }
.empty-p { font-size: 0.85rem; color: var(--muted); line-height: 1.7; max-width: 280px; margin: 0 auto 1.5rem; }
.empty-hint { display: inline-flex; align-items: center; gap: 6px; background: var(--blue-light); color: var(--blue); border: 1px solid #c5d0ff; padding: 7px 16px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; }
.feature-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.5rem; justify-content: center; }
.fchip { background: var(--white); border: 1px solid var(--border); border-radius: 8px; padding: 5px 12px; font-size: 0.72rem; font-weight: 600; color: var(--muted); display: flex; align-items: center; gap: 5px; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.footer-item { display: flex; align-items: center; gap: 5px; }
</style>
""", unsafe_allow_html=True)

# ─── Encoding Maps ────────────────────────────────────────────────────────────
FAT_CONTENT_MAP = {"Low Fat": 0, "Regular": 1}
ITEM_TYPE_MAP = {
    "Baking Goods": 0, "Breads": 1, "Breakfast": 2, "Canned": 3,
    "Dairy": 4, "Frozen Foods": 5, "Fruits and Vegetables": 6,
    "Hard Drinks": 7, "Health and Hygiene": 8, "Household": 9,
    "Meat": 10, "Others": 11, "Seafood": 12, "Snack Foods": 13,
    "Soft Drinks": 14, "Starchy Foods": 15,
}
OUTLET_SIZE_MAP  = {"Small": 0, "Medium": 1, "High": 2}
OUTLET_LOC_MAP   = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
OUTLET_TYPE_MAP  = {
    "Grocery Store": 0,
    "Supermarket Type 1": 1,
    "Supermarket Type 2": 2,
    "Supermarket Type 3": 3,
}
OUTLET_ID_MAP = {
    "OUT010": 0, "OUT013": 1, "OUT017": 2, "OUT018": 3, "OUT019": 4,
    "OUT027": 5, "OUT035": 6, "OUT045": 7, "OUT046": 8, "OUT049": 9,
}
MAX_SALES = 13000

# ─── Background ──────────────────────────────────────────────────────────────
st.markdown('<div class="bg-grid"></div>', unsafe_allow_html=True)

# ─── Navbar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-left">
    <div class="nav-logo">🛒</div>
    <div>
      <div class="nav-brand">Big<span>Mart</span> Analytics</div>
      <div class="nav-tagline">AI-Powered Sales Intelligence</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-pill blue">XGBoost Model</div>
    <div class="nav-pill green"><span class="live-dot"></span>Live</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-chip">🤖 Machine Learning · Retail Analytics</div>
    <h1>Predict <span>Item Outlet</span><br>Sales Instantly</h1>
    <p class="hero-desc">
      Enter product and store details to get an AI-powered sales estimate.
      Helps retailers forecast demand, optimize inventory, and drive smarter business decisions.
    </p>
    <div class="hero-stats">
      <div>
        <div class="hstat-val">8,523</div>
        <div class="hstat-label">Training Records</div>
      </div>
      <div class="hstat-div"></div>
      <div>
        <div class="hstat-val">16</div>
        <div class="hstat-label">Product Categories</div>
      </div>
      <div class="hstat-div"></div>
      <div>
        <div class="hstat-val">10</div>
        <div class="hstat-label">Outlets Covered</div>
      </div>
      <div class="hstat-div"></div>
      <div>
        <div class="hstat-val">11</div>
        <div class="hstat-label">Input Features</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── How it Works ────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-bar">
  <div class="how-step"><div class="how-num">1</div><div class="how-text">Select product category & details</div></div>
  <div class="how-step"><div class="how-num">2</div><div class="how-text">Set pricing & visibility info</div></div>
  <div class="how-step"><div class="how-num">3</div><div class="how-text">Choose your outlet & location</div></div>
  <div class="how-step"><div class="how-num">4</div><div class="how-text">Click Predict & view results</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:

    # Product Card
    st.markdown("""
    <div class="fcard">
      <div class="fcard-header">
        <div class="fcard-icon blue">📦</div>
        <div>
          <div class="fcard-title">Product Information</div>
          <div class="fcard-sub">Details about the item being predicted</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        item_type   = st.selectbox("Item Category", list(ITEM_TYPE_MAP.keys()))
        fat_content = st.selectbox("Fat Content", list(FAT_CONTENT_MAP.keys()))
    with col2:
        item_mrp    = st.number_input("Max Retail Price (₹)", min_value=10.0, max_value=270.0, value=150.0, step=0.5, format="%.2f")
        item_weight = st.number_input("Item Weight (kg)", min_value=1.0, max_value=25.0, value=12.0, step=0.01, format="%.2f")

    item_visibility = st.slider(
        "Shelf Visibility  (0.000 = hidden → 0.330 = fully visible)",
        min_value=0.0, max_value=0.33, value=0.05, step=0.001, format="%.3f"
    )

    # Outlet Card
    st.markdown("""
    <div class="fcard" style="margin-top:1.2rem;">
      <div class="fcard-header">
        <div class="fcard-icon teal">🏪</div>
        <div>
          <div class="fcard-title">Outlet / Store Information</div>
          <div class="fcard-sub">Details about the retail outlet</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        outlet_id   = st.selectbox("Outlet ID", list(OUTLET_ID_MAP.keys()))
        outlet_type = st.selectbox("Outlet Type", list(OUTLET_TYPE_MAP.keys()))
    with col4:
        outlet_size = st.selectbox("Outlet Size", list(OUTLET_SIZE_MAP.keys()))
        outlet_loc  = st.selectbox("Location Tier", list(OUTLET_LOC_MAP.keys()))

    outlet_year = st.selectbox("Establishment Year", options=list(range(1985, 2010)), index=14)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮  Predict Sales Now")


with right:

    if predict_clicked:
        with st.spinner("Running prediction model..."):
            time.sleep(0.6)

        features = np.array([[
            OUTLET_ID_MAP[outlet_id],
            item_weight,
            FAT_CONTENT_MAP[fat_content],
            item_visibility,
            ITEM_TYPE_MAP[item_type],
            item_mrp,
            OUTLET_ID_MAP[outlet_id],
            outlet_year,
            OUTLET_SIZE_MAP[outlet_size],
            OUTLET_LOC_MAP[outlet_loc],
            OUTLET_TYPE_MAP[outlet_type],
        ]])

        model_path = "bigmart_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            prediction = float(model.predict(features)[0])
            prediction = max(0.0, prediction)

            # Category
            if prediction >= 4500:
                cat, cat_cls, cat_icon = "High Sales",     "high",   "🟢"
            elif prediction >= 1800:
                cat, cat_cls, cat_icon = "Moderate Sales", "medium", "🟡"
            else:
                cat, cat_cls, cat_icon = "Low Sales",      "low",    "🔴"

            # Labels
            price_tier  = "Budget" if item_mrp < 80 else ("Mid-Range" if item_mrp < 180 else "Premium")
            vis_label   = "Poor"   if item_visibility < 0.05 else ("Average" if item_visibility < 0.15 else "High")
            outlet_age  = 2024 - outlet_year
            meter_pct   = min(100, int((prediction / MAX_SALES) * 100))
            meter_color = "#059669" if cat_cls == "high" else ("#d97706" if cat_cls == "medium" else "#dc2626")

            reco_map = {
                "low":    "📉 <strong>Low sales alert.</strong> Consider boosting shelf visibility, reducing the price point, or moving this product to a higher-traffic Supermarket outlet for better exposure.",
                "medium": "📊 <strong>Moderate sales.</strong> This product shows growth potential. Improving visibility or upgrading to a Tier 1 Supermarket Type 2 outlet could significantly increase performance.",
                "high":   "🚀 <strong>Strong sales predicted!</strong> This product is performing excellently. Maintain current pricing and shelf placement — consider increasing stock to meet demand.",
            }
            reco = reco_map[cat_cls]

            # Result card
            st.markdown(f"""
            <div class="res-card {cat_cls}">
              <div class="res-eyebrow">Predicted Item Outlet Sales</div>
              <div class="res-amount"><span class="res-currency">₹</span>{prediction:,.0f}</div>
              <div class="res-sub">Estimated sales for this product at the selected outlet</div>
              <div class="res-badge">{cat_icon} {cat}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metric pills
            st.markdown(f"""
            <div class="metrics-row">
              <div class="metric-pill">
                <div class="mp-icon">💰</div>
                <div class="mp-val">{price_tier}</div>
                <div class="mp-label">Price Tier</div>
              </div>
              <div class="metric-pill">
                <div class="mp-icon">👁️</div>
                <div class="mp-val">{vis_label}</div>
                <div class="mp-label">Visibility</div>
              </div>
              <div class="metric-pill">
                <div class="mp-icon">🏗️</div>
                <div class="mp-val">{outlet_age} yrs</div>
                <div class="mp-label">Outlet Age</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Meter
            st.markdown(f"""
            <div class="meter-card">
              <div class="meter-header">
                <div class="meter-title">Sales Performance Meter</div>
                <div class="meter-pct">{meter_pct}% of max range</div>
              </div>
              <div class="meter-track">
                <div class="meter-fill" style="width:{meter_pct}%; background: linear-gradient(90deg, {meter_color}99, {meter_color});"></div>
              </div>
              <div class="meter-labels"><span>₹0</span><span>₹6,500</span><span>₹13,000</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Summary
            rows = [
                ("📦 Category",      item_type),
                ("🧴 Fat Content",   fat_content),
                ("⚖️ Weight",        f"{item_weight} kg"),
                ("💵 MRP",           f"₹{item_mrp:.2f}"),
                ("👁️ Visibility",    f"{item_visibility:.3f}"),
                ("🏪 Outlet",        outlet_id),
                ("🏬 Type",          outlet_type.replace("Supermarket", "SM")),
                ("📐 Size",          outlet_size),
                ("📍 Location",      outlet_loc),
                ("📅 Est. Year",     str(outlet_year)),
            ]
            rows_html = "".join(
                f'<div class="sum-row"><span class="sum-key">{k}</span><span class="sum-val">{v}</span></div>'
                for k, v in rows
            )
            st.markdown(f"""
            <div class="sum-card">
              <div class="sum-title">📋 Input Summary</div>
              {rows_html}
            </div>
            <div class="reco-card">
              <div class="reco-icon">💡</div>
              <div class="reco-text">{reco}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("⚠️ `bigmart_model.pkl` not found. Place it in the same folder as `app.py` and restart.")

    else:
        st.markdown("""
        <div class="empty-wrap">
          <span class="empty-emoji">🔮</span>
          <div class="empty-h">Your Prediction Appears Here</div>
          <p class="empty-p">
            Fill in the product and outlet details on the left panel,
            then click <strong>Predict Sales Now</strong> to see the AI-powered result.
          </p>
          <div class="empty-hint">← Start by filling the form</div>
          <div class="feature-chips">
            <div class="fchip">📊 Sales Category</div>
            <div class="fchip">💰 Price Tier</div>
            <div class="fchip">📈 Performance Meter</div>
            <div class="fchip">💡 Smart Recommendation</div>
            <div class="fchip">📋 Input Summary</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-item">🛒 BigMart Sales Predictor</div>
  <div class="footer-item">🤖 XGBoost Regression Model</div>
  <div class="footer-item">⚡ Built with Streamlit</div>
  <div class="footer-item">🎓 ML Internship Project</div>
</div>
""", unsafe_allow_html=True)