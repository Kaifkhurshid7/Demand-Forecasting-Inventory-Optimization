"""
Streamlit Dashboard — Main Entry Point.

Provides multi-page navigation for the Demand Forecasting & Inventory
Optimization system.
"""

import streamlit as st
import pandas as pd
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.config import PROJECT_ROOT, TARGET_COL, CATEGORY_COL

logging.basicConfig(level=logging.INFO)

# ── Page Config ──
st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Data (cached) ──

@st.cache_data
def load_daily_data():
    """Load processed daily category demand data."""
    from src.data_loader import load_processed
    try:
        return load_processed("daily_category_demand")
    except FileNotFoundError:
        st.error("Processed data not found. Run `python src/data_loader.py` first.")
        return pd.DataFrame()

@st.cache_data
def load_featured_data():
    """Load featured data if available."""
    try:
        from src.utils import load_model
        return load_model("featured_data.pkl")
    except (FileNotFoundError, AttributeError):
        return pd.DataFrame()

# ── Initialize session state ──
if "daily_data" not in st.session_state:
    st.session_state.daily_data = load_daily_data()
if "featured_data" not in st.session_state:
    st.session_state.featured_data = load_featured_data()

daily = st.session_state.daily_data
featured = st.session_state.featured_data


# ── Sidebar Navigation ──

st.sidebar.markdown("# 📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📈 EDA Overview",
        "🤖 Model Comparison",
        "🔮 Forecast Viewer",
        "📦 Inventory Optimizer",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Data status
st.sidebar.markdown("### Data Status")
if not daily.empty:
    n_cats = daily[CATEGORY_COL].nunique() if CATEGORY_COL in daily.columns else 0
    st.sidebar.success(f"✅ {len(daily):,} rows\n{n_cats} categories")
else:
    st.sidebar.warning("⚠️ No data loaded")

if not featured.empty:
    st.sidebar.info(f"✅ Featured data: {len(featured):,} rows")
else:
    st.sidebar.warning("⚠️ No featured data")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly")


# ── Page Routing ──

if page == "🏠 Home":
    st.title("📊 Demand Forecasting & Inventory Optimization")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Categories Tracked", daily[CATEGORY_COL].nunique() if not daily.empty else 0)
    with col2:
        total_orders = int(daily[TARGET_COL].sum()) if not daily.empty else 0
        st.metric("Total Orders", f"{total_orders:,}")
    with col3:
        date_range = f"{daily['date'].min().date()} → {daily['date'].max().date()}" if not daily.empty else "N/A"
        st.metric("Date Range", date_range)

    st.markdown("""
    ### Overview

    This end-to-end system forecasts daily order volume across product categories
    using the **Brazilian E-Commerce (Olist)** dataset. It trains **Prophet**,
    **XGBoost**, **LSTM**, and an **Ensemble** model, then uses **PuLP** for
    inventory optimization.

    ### Pipeline
    1. **Data Ingestion** — Load & clean 9 CSV files
    2. **EDA** — Visualize trends, seasonality, category patterns
    3. **Feature Engineering** — Lags, rolling windows, seasonal dummies
    4. **Model Training** — Prophet, XGBoost, LSTM, Weighted Ensemble
    5. **Inventory Optimization** — LP solver for reorder quantities
    6. **API** — FastAPI for inference
    7. **Dashboard** — Interactive exploration (you are here)

    ### Quick Start
    ```bash
    # Train models
    python src/train.py

    # Launch API
    uvicorn api.main:app --reload

    # Launch dashboard
    streamlit run dashboard/app.py
    ```
    """)

elif page == "📈 EDA Overview":
    exec(open("dashboard/pages/01_eda.py", encoding="utf-8").read())

elif page == "🤖 Model Comparison":
    exec(open("dashboard/pages/02_model_comparison.py", encoding="utf-8").read())

elif page == "🔮 Forecast Viewer":
    exec(open("dashboard/pages/03_forecast.py", encoding="utf-8").read())

elif page == "📦 Inventory Optimizer":
    exec(open("dashboard/pages/04_optimization.py", encoding="utf-8").read())
