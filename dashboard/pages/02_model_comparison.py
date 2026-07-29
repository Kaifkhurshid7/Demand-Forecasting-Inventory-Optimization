"""
Model Comparison Page — Compare forecast performance across models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR

st.set_page_config(layout="wide", page_title="Model Comparison")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("[*] Model Comparison")
st.markdown("---")


# -- Load Metrics --
@st.cache_data
def load_saved_metrics():
    metrics_file = MODELS_DIR / "all_categories_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {}

saved_metrics = load_saved_metrics()


# -- Check model availability --
model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
categories_available = sorted(set(
    f.stem.replace("ensemble_", "") for f in model_files
))

if not categories_available:
    st.warning("No trained models found. Please run `python src/train.py` first.")
    st.info("""
    The training pipeline produces ensemble models for each product category.

    ```bash
    # Train all models
    python src/train.py
    ```
    """)
    st.stop()

st.success(f"[OK] Models loaded for {len(categories_available)} categories")


# -- Aggregate Metrics --
if saved_metrics:
    st.subheader("[+] Aggregate Performance Summary")

    metrics_df = pd.DataFrame.from_dict(saved_metrics, orient="index")
    metrics_df.index.name = "category"
    metrics_df = metrics_df.reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average MAPE", f"{metrics_df['mape'].mean():.2f}%")
    with col2:
        st.metric("Average RMSE", f"{metrics_df['rmse'].mean():.2f}")
    with col3:
        st.metric("Average MAE", f"{metrics_df['mae'].mean():.2f}")

    # Distribution of MAPE
    fig = px.histogram(
        metrics_df,
        x="mape",
        nbins=20,
        title="Distribution of MAPE Across Categories",
        labels={"mape": "MAPE (%)"},
        template="plotly_white",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_vline(x=15, line_dash="dash", line_color="red",
                  annotation_text="Target (<15%)")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Category-Level Metrics")
    metrics_df_sorted = metrics_df.sort_values("mape")
    st.dataframe(
        metrics_df_sorted.style
        .format({"mape": "{:.2f}%", "rmse": "{:.2f}", "mae": "{:.2f}"})
        .background_gradient(subset=["mape"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )


# -- Per-Category Forecast vs Actual --
st.subheader("[Z] Per-Category Forecast vs Actual")

selected_cat = st.selectbox("Select category to examine", categories_available)

if selected_cat and not featured.empty:
    cat_data = featured[featured[CATEGORY_COL] == selected_cat].sort_values("date")

    # Use test period (last ~60 days)
    test_data = cat_data.tail(60).copy()
    if len(test_data) > 0:
        actual = test_data[TARGET_COL].values
        dates = test_data["date"].values

        # Try to get predictions -- these would ideally be stored
        # For now, show actual vs historical rolling mean as baseline
        baseline = test_data[TARGET_COL].shift(1).rolling(7, min_periods=1).mean().values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=baseline,
            mode="lines",
            name="7-day MA Baseline",
            line=dict(color="gray", width=1.5, dash="dash"),
        ))

        fig.update_layout(
            title=f"Recent Demand — {selected_cat}",
            xaxis_title="Date",
            yaxis_title="Daily Orders",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model error metrics
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(actual[7:], baseline[7:])
        st.caption(f"Baseline (7-day MA) MAE: {mae:.2f}")

    # Category statistics
    st.subheader("[#] Category Statistics")
    stats = cat_data[TARGET_COL].describe()
    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
        "Value": [
            f"{stats['mean']:.1f}",
            f"{stats['std']:.1f}",
            f"{stats['min']:.0f}",
            f"{stats['25%']:.0f}",
            f"{stats['50%']:.0f}",
            f"{stats['75%']:.0f}",
            f"{stats['max']:.0f}",
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)


# -- Model Architecture Summary --
st.subheader("[i] Model Architecture Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(
        "### Prophet\n"
        "**Additive model** with yearly and weekly seasonality.\n\n"
        "Handles holidays natively as regressors.\n\n"
        "Best for: capturing trend + strong seasonality."
    )

with col2:
    st.info(
        "### XGBoost\n"
        "**Gradient boosting** on tabular features.\n\n"
        "Uses lags, rolling stats, seasonal dummies.\n\n"
        "Best for: feature-rich demand patterns."
    )

with col3:
    st.info(
        "### LSTM\n"
        "**Recurrent neural network** (2-layer).\n\n"
        "Sliding window of 30 days.\n\n"
        "Best for: long-range temporal dependencies."
    )

with col4:
    st.info(
        "### Ensemble\n"
        "**Weighted average** of all models.\n\n"
        "Weights optimized via validation RMSE.\n\n"
        "Best for: robust, stable forecasts."
    )
