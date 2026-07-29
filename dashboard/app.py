"""
Streamlit Dashboard — Main Entry Point.
Dark theme professional design for Demand Forecasting & Inventory Optimization.
"""

import streamlit as st
import pandas as pd
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.config import TARGET_COL, CATEGORY_COL

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DARK THEME CSS
# ============================================================
st.markdown("""
<style>
    /* Base dark theme */
    .stApp {
        background-color: #0D1117 !important;
        color: #C9D1D9 !important;
    }
    .main > div {
        background-color: #0D1117;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif !important;
        color: #F0F6FC !important;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.25rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.75rem !important;
    }
    p, li, .stMarkdown, .stMarkdown p {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #C9D1D9 !important;
    }

    /* Links */
    a {
        color: #58A6FF !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D1117 !important;
        min-width: 260px !important;
        max-width: 260px !important;
        border-right: 1px solid #21262D;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8B949E !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #21262D;
        margin: 1rem 0;
    }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 1.5rem 1rem 0.75rem;
        border-bottom: 1px solid #21262D;
        margin-bottom: 0.5rem;
    }
    .sidebar-brand h2 {
        color: #F0F6FC !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .sub {
        color: #8B949E !important;
        font-size: 0.72rem !important;
        margin: 0.1rem 0 0 0 !important;
    }

    /* Sidebar nav buttons */
    div.stButton > button {
        background: transparent !important;
        color: #8B949E !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.55rem 0.75rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        text-align: left !important;
        transition: all 0.12s ease !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background: #161B22 !important;
        color: #C9D1D9 !important;
    }
    div.stButton > button:focus {
        background: #1F2937 !important;
        color: #58A6FF !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        border: none !important;
    }
    div.stButton > button:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    div.stButton > button:active {
        background: #1F2937 !important;
        color: #58A6FF !important;
    }
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: transparent !important;
    }

    /* Sidebar nav active state */
    .nav-active {
        background: #1F2937 !important;
        color: #58A6FF !important;
        font-weight: 600 !important;
        border-radius: 6px;
    }

    /* Sidebar section labels */
    .sidebar-label {
        font-size: 0.65rem !important;
        color: #484F58 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0 0.75rem;
        margin-bottom: 0.15rem;
    }

    /* Metric Cards - Dark */
    div[data-testid="metric-container"] {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 8px !important;
        padding: 0.85rem 1.1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        transition: border-color 0.15s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #30363D !important;
    }
    div[data-testid="metric-container"] > div:first-child {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        color: #8B949E !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #F0F6FC !important;
        margin-top: 0.1rem;
    }

    /* Content area padding */
    .main .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Card component - Dark */
    .card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: border-color 0.15s;
    }
    .card:hover {
        border-color: #30363D;
    }
    .card h3 {
        margin-top: 0 !important;
        font-size: 0.9rem !important;
        color: #F0F6FC !important;
        font-weight: 600;
    }
    .card p {
        font-size: 0.82rem;
        color: #8B949E;
        margin: 0.3rem 0 0;
    }
    .card .tag {
        display: inline-block;
        padding: 0.12rem 0.45rem;
        border-radius: 9999px;
        font-size: 0.65rem;
        font-weight: 600;
        background: #1F2937;
        color: #58A6FF;
    }

    /* Section header with accent bar */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.25rem 0 0.75rem;
    }
    .section-header .bar {
        width: 3px;
        height: 1rem;
        background: #58A6FF;
        border-radius: 2px;
        flex-shrink: 0;
    }
    .section-header h2 {
        margin: 0 !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #21262D;
        margin: 1.25rem 0;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: #1F6FEB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        box-shadow: 0 1px 3px rgba(31, 111, 235, 0.2);
        transition: all 0.15s;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #388BFD !important;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #21262D;
    }
    .stDataFrame table {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace !important;
        font-size: 0.78rem !important;
    }
    .stDataFrame thead tr th {
        background: #161B22 !important;
        color: #8B949E !important;
        font-weight: 600 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.55rem 0.75rem !important;
        border-bottom: 1px solid #21262D !important;
    }
    .stDataFrame tbody tr td {
        color: #C9D1D9 !important;
        border-bottom: 1px solid #21262D !important;
        padding: 0.45rem 0.75rem !important;
    }
    .stDataFrame tbody tr:hover {
        background: #1C2128 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: #C9D1D9 !important;
    }
    .streamlit-expanderContent {
        background: #0D1117 !important;
        border: 1px solid #21262D !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 500;
        color: #8B949E;
        padding: 0.5rem 0.75rem;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #F0F6FC !important;
        background: #161B22 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #58A6FF !important;
    }

    /* Select / Input */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #8B949E !important;
    }
    div[data-baseweb="select"] > div {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 6px !important;
        color: #C9D1D9 !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #30363D !important;
    }
    .stSlider > div > div {
        color: #58A6FF !important;
    }
    .stNumberInput input {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 6px !important;
        color: #C9D1D9 !important;
    }

    /* Status messages - dark */
    .stAlert > div {
        border-radius: 6px !important;
        border: none !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stSuccess"] > div {
        background: #0C2D1B !important;
        color: #3FB950 !important;
        border: 1px solid #1A3F2B !important;
    }
    div[data-testid="stWarning"] > div {
        background: #2D1C0C !important;
        color: #D29922 !important;
        border: 1px solid #3D2E0C !important;
    }
    div[data-testid="stError"] > div {
        background: #2D0C0C !important;
        color: #F85149 !important;
        border: 1px solid #3D1C1C !important;
    }
    div[data-testid="stInfo"] > div {
        background: #0C1C2D !important;
        color: #58A6FF !important;
        border: 1px solid #1C2D3D !important;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.12rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-blue {
        background: #1F2937;
        color: #58A6FF;
    }
    .badge-green {
        background: #0C2D1B;
        color: #3FB950;
    }

    /* Checkbox */
    .stCheckbox label {
        color: #C9D1D9 !important;
        font-size: 0.82rem !important;
    }

    /* Code blocks */
    pre {
        background: #0D1117 !important;
        border: 1px solid #21262D !important;
        border-radius: 6px !important;
        color: #C9D1D9 !important;
    }
    code {
        color: #FFA657 !important;
    }

    /* Sidebar footer */
    .sidebar-footer p {
        font-size: 0.65rem !important;
        color: #484F58 !important;
        margin: 0 !important;
    }

    /* Download button */
    button[title="Download"] {
        background: #21262D !important;
        color: #C9D1D9 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    button[title="Download"]:hover {
        background: #30363D !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #58A6FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_daily_data():
    from src.data_loader import load_processed
    try:
        return load_processed("daily_category_demand")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_featured_data():
    from src.utils import load_model
    try:
        return load_model("featured_data.pkl")
    except (FileNotFoundError, AttributeError):
        return pd.DataFrame()

if "daily_data" not in st.session_state:
    st.session_state.daily_data = load_daily_data()
if "featured_data" not in st.session_state:
    st.session_state.featured_data = load_featured_data()
if "page" not in st.session_state:
    st.session_state.page = "Home"

daily = st.session_state.daily_data
featured = st.session_state.featured_data


# ============================================================
# SIDEBAR
# ============================================================
def set_page(p):
    st.session_state.page = p

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>Demand Forecasting</h2>
        <p class="sub">Inventory Optimization System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)

    nav_items = {
        "Home": "+",
        "EDA Overview": "~",
        "Model Comparison": "*",
        "Forecast Viewer": ">",
        "Inventory Optimizer": "$",
    }

    current = st.session_state.page
    for label, icon in nav_items.items():
        active = "nav-active" if current == label else ""
        if st.button(f"  {icon}   {label}", key=f"nav_{label}"):
            set_page(label)

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Data Status</p>', unsafe_allow_html=True)

    if not daily.empty:
        n_cats = daily[CATEGORY_COL].nunique() if CATEGORY_COL in daily.columns else 0
        st.markdown(
            f"<div style='background:#0C2D1B; color:#3FB950; padding:0.45rem 0.75rem; "
            f"border-radius:6px; font-size:0.78rem; margin:0 0 0.35rem; "
            f"border:1px solid #1A3F2B;'>"
            f"OK  {len(daily):,} rows  |  {n_cats} categories</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#2D1C0C; color:#D29922; padding:0.45rem 0.75rem; "
            "border-radius:6px; font-size:0.78rem; margin:0 0 0.35rem; "
            "border:1px solid #3D2E0C;'>"
            "!  No data loaded</div>",
            unsafe_allow_html=True
        )

    if not featured.empty:
        st.markdown(
            f"<div style='background:#0C1C2D; color:#58A6FF; padding:0.45rem 0.75rem; "
            f"border-radius:6px; font-size:0.78rem; border:1px solid #1C2D3D;'>"
            f"i  Featured: {len(featured):,} rows</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#2D1C0C; color:#D29922; padding:0.45rem 0.75rem; "
            "border-radius:6px; font-size:0.78rem; border:1px solid #3D2E0C;'>"
            "!  No featured data</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer"><p>Kaif Khurshid  |  XIM University<br>'
        'Built with Streamlit + Plotly</p></div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE ROUTING
# ============================================================
page = st.session_state.page

if page == "Home":
    st.title("Demand Forecasting & Inventory Optimization")
    st.markdown(
        "<p style='color:#8B949E; font-size:0.85rem; margin-top:-0.25rem;'>"
        "Kaif Khurshid  |  XIM University  |  B.Tech Final Year (2023-27)</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Categories Tracked", daily[CATEGORY_COL].nunique() if not daily.empty else 0)
    with col2:
        total_orders = int(daily[TARGET_COL].sum()) if not daily.empty else 0
        st.metric("Total Orders", f"{total_orders:,}")
    with col3:
        avg_daily = int(daily.groupby("date")[TARGET_COL].sum().mean()) if not daily.empty else 0
        st.metric("Avg Daily Orders", f"{avg_daily:,}")
    with col4:
        date_range = f"{daily['date'].min().date()} - {daily['date'].max().date()}" if not daily.empty else "N/A"
        st.metric("Date Range", date_range)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>System Overview</h3>
            <p>
            This system forecasts daily order volume across product categories
            using the <strong>Brazilian E-Commerce (Olist)</strong> dataset. It trains <strong>Prophet</strong>,
            <strong>XGBoost</strong>, <strong>LSTM</strong>, and an <strong>Ensemble</strong> model, then uses <strong>PuLP</strong>
            linear programming for inventory optimization.
            </p>
            <br>
            <h3>Pipeline Stages</h3>
            <ol style="padding-left:1.2rem; margin-top:0.3rem;">
                <li><strong>Data Ingestion</strong> - Load and clean 9 CSV files</li>
                <li><strong>EDA</strong> - Visualize trends and category patterns</li>
                <li><strong>Feature Engineering</strong> - Lags, rolling windows, dummies</li>
                <li><strong>Model Training</strong> - Prophet, XGBoost, LSTM, Ensemble</li>
                <li><strong>Inventory Optimization</strong> - LP solver for reorder qty</li>
                <li><strong>API & Dashboard</strong> - FastAPI + Streamlit</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>Quick Start</h3>
            <pre style="background:#0D1117; padding:0.75rem; border-radius:6px; font-size:0.78rem; margin-top:0.5rem; overflow-x:auto;">
# Train models
python src/train.py

# Launch API
uvicorn api.main:app --reload

# Launch dashboard
streamlit run dashboard/app.py
            </pre>
            <div style="margin-top:0.75rem;">
                <span class="badge badge-blue">API: localhost:8000</span>
                <span class="badge badge-green" style="margin-left:0.4rem;">Dashboard: localhost:8501</span>
            </div>
            <br>
            <h3>Model Performance Target</h3>
            <p>MAPE < 15%  |  RMSE minimized  |  Service Level >= 95%</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "EDA Overview":
    exec(open("dashboard/pages/01_eda.py", encoding="utf-8").read())

elif page == "Model Comparison":
    exec(open("dashboard/pages/02_model_comparison.py", encoding="utf-8").read())

elif page == "Forecast Viewer":
    exec(open("dashboard/pages/03_forecast.py", encoding="utf-8").read())

elif page == "Inventory Optimizer":
    exec(open("dashboard/pages/04_optimization.py", encoding="utf-8").read())
