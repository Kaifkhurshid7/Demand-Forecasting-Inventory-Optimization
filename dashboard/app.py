"""
Streamlit Dashboard — Main Entry Point.
Professional SaaS-style design for Demand Forecasting & Inventory Optimization.
"""

import streamlit as st
import pandas as pd
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.config import TARGET_COL, CATEGORY_COL

logging.basicConfig(level=logging.INFO)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon=str(Path(__file__).parent / "favicon.ico") if (Path(__file__).parent / "favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Professional SaaS Design System
# ============================================================
st.markdown("""
<style>
    /* ── Base Reset ── */
    #root > div:first-child {
        padding: 0 !important;
    }
    .main > div {
        padding: 0 !important;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.25rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    p, li, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #334155;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        min-width: 260px !important;
        max-width: 260px !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #CBD5E1;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F1F5F9;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E293B;
        margin: 1rem 0;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #94A3B8 !important;
        font-size: 0.875rem;
        font-weight: 500;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        transition: all 0.15s ease;
        width: 100%;
        display: block;
    }
    section[data-testid="stSidebar"] .stRadio > label:hover {
        background-color: #1E293B;
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label[data-checked="true"] {
        background-color: #1E3A5F !important;
        color: #60A5FA !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] > p {
        color: #94A3B8;
        font-size: 0.8rem;
    }

    /* ── Sidebar Brand ── */
    .sidebar-brand {
        padding: 1.25rem 1rem 0.5rem;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 0.75rem;
    }
    .sidebar-brand h2 {
        color: #F1F5F9 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand p {
        color: #64748B !important;
        font-size: 0.72rem !important;
        margin: 0.15rem 0 0 0 !important;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        padding: 1rem;
        border-top: 1px solid #1E293B;
        width: 259px;
        background-color: #0F172A;
    }
    .sidebar-footer p {
        font-size: 0.7rem !important;
        color: #475569 !important;
        margin: 0 !important;
    }

    /* ── Metric Cards ── */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] > div:first-child {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-top: 0.15rem;
    }

    /* ── Content Area ── */
    .main .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* ── Cards / Containers ── */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .saas-card h3 {
        margin-top: 0 !important;
        font-size: 0.95rem !important;
        color: #0F172A;
    }
    .saas-card .subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.25rem 0 0.75rem;
    }
    .section-header .accent-bar {
        width: 3px;
        height: 1.1rem;
        background: #3B82F6;
        border-radius: 2px;
        flex-shrink: 0;
    }
    .section-header h2 {
        margin: 0 !important;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #1E3A5F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(30, 58, 95, 0.2) !important;
    }
    .stButton > button:hover {
        background: #152C4A !important;
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3) !important;
        transform: translateY(-1px);
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        overflow: hidden;
    }
    .stDataFrame table {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace !important;
        font-size: 0.8rem !important;
    }
    .stDataFrame thead tr th {
        background: #F8FAFC !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.6rem 0.75rem !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #F8FAFC !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #1E293B !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #E2E8F0 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1E3A5F !important;
        font-weight: 600 !important;
    }

    /* ── Select / Input ── */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ── Status Messages ── */
    .stAlert > div {
        border-radius: 8px !important;
        border: none !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stSuccess"] > div {
        background: #ECFDF5 !important;
        color: #065F46 !important;
    }
    div[data-testid="stWarning"] > div {
        background: #FFFBEB !important;
        color: #92400E !important;
    }
    div[data-testid="stError"] > div {
        background: #FEF2F2 !important;
        color: #991B1B !important;
    }
    div[data-testid="stInfo"] > div {
        background: #EFF6FF !important;
        color: #1E40AF !important;
    }

    /* ── Sidebar Nav Spacing ── */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0.75rem;
        margin: 0.1rem 0;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        cursor: pointer;
        transition: all 0.15s;
    }
    .nav-item:hover {
        background: #1E293B;
        color: #E2E8F0;
    }
    .nav-item.active {
        background: #1E3A5F;
        color: #60A5FA;
        font-weight: 600;
    }
    .nav-icon {
        width: 1.4rem;
        text-align: center;
        font-size: 1rem;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-blue {
        background: #DBEAFE;
        color: #1E40AF;
    }
    .badge-green {
        background: #D1FAE5;
        color: #065F46;
    }
    .badge-red {
        background: #FEE2E2;
        color: #991B1B;
    }
    .badge-gray {
        background: #F1F5F9;
        color: #475569;
    }

    /* ── Tooltip text helper ── */
    .text-muted {
        color: #64748B;
        font-size: 0.8rem;
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
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <h2>Demand Forecasting</h2>
        <p>Inventory Optimization System</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("<p style='font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.06em; padding:0 0.75rem; margin-bottom:0.25rem;'>Navigation</p>", unsafe_allow_html=True)

    nav_items = {
        "Home": "[+]",
        "EDA Overview": "[~]",
        "Model Comparison": "[*]",
        "Forecast Viewer": "[>]",
        "Inventory Optimizer": "[$]",
    }

    for label, icon in nav_items.items():
        if st.sidebar.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.page = label

    # Data Status
    st.markdown("---")
    st.markdown("<p style='font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.06em; padding:0 0.75rem; margin-bottom:0.25rem;'>Data Status</p>", unsafe_allow_html=True)

    if not daily.empty:
        n_cats = daily[CATEGORY_COL].nunique() if CATEGORY_COL in daily.columns else 0
        st.markdown(
            f"<div style='background:#065F46; color:#D1FAE5; padding:0.5rem 0.75rem; "
            f"border-radius:6px; font-size:0.8rem; margin:0 0 0.4rem;'>"
            f"[OK] {len(daily):,} rows  |  {n_cats} categories</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#78350F; color:#FDE68A; padding:0.5rem 0.75rem; "
            "border-radius:6px; font-size:0.8rem; margin:0 0 0.4rem;'>"
            "[!] No data loaded</div>",
            unsafe_allow_html=True
        )

    if not featured.empty:
        st.markdown(
            f"<div style='background:#1E3A5F; color:#DBEAFE; padding:0.5rem 0.75rem; "
            f"border-radius:6px; font-size:0.8rem;'>"
            f"[i] Featured: {len(featured):,} rows</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#78350F; color:#FDE68A; padding:0.5rem 0.75rem; "
            "border-radius:6px; font-size:0.8rem;'>"
            "[!] No featured data</div>",
            unsafe_allow_html=True
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem; color:#475569; padding:0 0.75rem;'>"
        "Kaif Khurshid  |  XIM University<br>"
        "Built with Streamlit + Plotly</p>",
        unsafe_allow_html=True
    )

# ============================================================
# PAGE ROUTING
# ============================================================
page = st.session_state.page

if page == "Home":
    # ── Home Page ──
    st.title("Demand Forecasting & Inventory Optimization")
    st.markdown(
        "<p style='color:#64748B; font-size:0.9rem; margin-top:-0.25rem;'>"
        "Kaif Khurshid  |  XIM University  |  B.Tech Final Year (2023-27)</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Metric cards
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

    # Overview card
    st.markdown("""
    <div class="saas-card">
        <h3>System Overview</h3>
        <p style="margin-top:0.5rem;">
        This end-to-end system forecasts daily order volume across product categories
        using the <strong>Brazilian E-Commerce (Olist)</strong> dataset. It trains <strong>Prophet</strong>,
        <strong>XGBoost</strong>, <strong>LSTM</strong>, and an <strong>Ensemble</strong> model, then uses <strong>PuLP</strong> linear
        programming for inventory optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="saas-card">
            <h3>Pipeline Stages</h3>
            <ol style="padding-left:1.2rem; margin-top:0.5rem;">
                <li><strong>Data Ingestion</strong> — Load and clean 9 CSV files</li>
                <li><strong>EDA</strong> — Visualize trends, seasonality, category patterns</li>
                <li><strong>Feature Engineering</strong> — Lags, rolling windows, seasonal dummies</li>
                <li><strong>Model Training</strong> — Prophet, XGBoost, LSTM, Weighted Ensemble</li>
                <li><strong>Inventory Optimization</strong> — LP solver for reorder quantities</li>
                <li><strong>API</strong> — FastAPI for inference</li>
                <li><strong>Dashboard</strong> — Interactive exploration (you are here)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="saas-card">
            <h3>Quick Start</h3>
            <pre style="background:#F1F5F9; padding:0.75rem; border-radius:6px; font-size:0.8rem; margin-top:0.5rem; overflow-x:auto;">
# Train models
python src/train.py

# Launch API
uvicorn api.main:app --reload

# Launch dashboard
streamlit run dashboard/app.py
            </pre>
            <div style="margin-top:0.75rem;">
                <span class="badge badge-green">API: localhost:8000</span>
                <span class="badge badge-blue" style="margin-left:0.4rem;">Dashboard: localhost:8501</span>
            </div>
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
