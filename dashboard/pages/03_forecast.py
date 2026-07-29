"""
Forecast Viewer Page — Select category and view future demand forecasts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR
from src.utils import load_model
from src.predict import predict as generate_forecast

st.set_page_config(layout="wide", page_title="Forecast Viewer")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("🔮 Forecast Viewer")
st.markdown("---")

# ── Discover available categories ──
model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
categories_available = sorted(set(
    f.stem.replace("ensemble_", "") for f in model_files
))

if not categories_available:
    st.warning("No trained models found. Please run `python src/train.py` first.")
    st.stop()

st.success(f"✅ {len(categories_available)} categories available for forecasting")

# ── Controls ──
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_cat = st.selectbox(
        "Select Category",
        categories_available,
        index=categories_available.index("bed_bath_table") if "bed_bath_table" in categories_available else 0,
    )
with col2:
    forecast_days = st.selectbox(
        "Forecast Horizon",
        [7, 14, 30, 60, 90],
        index=2,  # default 30
    )
with col3:
    show_individual = st.checkbox("Show individual model predictions", value=False)


# ── Generate Forecast ──
@st.cache_data(ttl=300)
def get_forecast(category: str, days: int):
    try:
        ensemble = load_model(f"ensemble_{category}.pkl")
        featured_data = load_model("featured_data.pkl")
        forecast_df, info = generate_forecast(
            category=category,
            days=days,
            results=ensemble,
            featured_data=featured_data,
        )
        return forecast_df, info
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return None, None


with st.spinner(f"Generating {forecast_days}-day forecast for {selected_cat}..."):
    forecast_df, info = get_forecast(selected_cat, forecast_days)

if forecast_df is None:
    st.stop()

# ── Summary Cards ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Forecast Total", f"{info['total_forecast_orders']:,} orders")
with col2:
    st.metric("Avg Daily", f"{info['avg_daily_forecast']:.1f}")
with col3:
    st.metric("Historical Avg", f"{info['avg_historical_daily']:.1f}")
with col4:
    change = ((info['avg_daily_forecast'] - info['avg_historical_daily'])
              / max(info['avg_historical_daily'], 0.01) * 100)
    st.metric("Trend vs History", f"{change:+.1f}%")


# ── Forecast Plot ──
st.subheader("📈 Forecast Visualization")

# Get historical data for context
if not daily.empty:
    hist_data = daily[daily[CATEGORY_COL] == selected_cat].sort_values("date")
    # Last 90 days of history
    hist_data = hist_data.tail(90)
else:
    hist_data = pd.DataFrame()

fig = go.Figure()

# Historical actuals
if not hist_data.empty:
    fig.add_trace(go.Scatter(
        x=hist_data["date"],
        y=hist_data[TARGET_COL],
        mode="lines",
        name="Historical Actual",
        line=dict(color="rgba(0,0,0,0.5)", width=1.5),
    ))

# Forecast with confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([pd.Series([hist_data["date"].iloc[-1]]) if not hist_data.empty else pd.Series([]),
                 forecast_df["date"]]),
    y=pd.concat([pd.Series([hist_data[TARGET_COL].iloc[-1]]) if not hist_data.empty else pd.Series([]),
                 forecast_df["predicted_upper"]]),
    mode="lines",
    name="Upper Bound",
    line=dict(width=0),
    showlegend=False,
    fill=None,
))
fig.add_trace(go.Scatter(
    x=pd.concat([pd.Series([hist_data["date"].iloc[-1]]) if not hist_data.empty else pd.Series([]),
                 forecast_df["date"]]),
    y=pd.concat([pd.Series([hist_data[TARGET_COL].iloc[-1]]) if not hist_data.empty else pd.Series([]),
                 forecast_df["predicted_lower"]]),
    mode="lines",
    name="Lower Bound",
    line=dict(width=0),
    fillcolor="rgba(46, 134, 171, 0.2)",
    fill="tonexty",
    showlegend=True,
))

# Mean forecast
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["predicted_orders"],
    mode="lines+markers",
    name="Forecast (Ensemble)",
    line=dict(color="#2E86AB", width=3),
    marker=dict(size=6),
))

# Individual model predictions
if show_individual:
    model_colors = {"prophet": "#E67E22", "xgboost": "#27AE60", "lstm": "#8E44AD"}
    for model_name, color in model_colors.items():
        col_name = f"{model_name}_prediction"
        if col_name in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[col_name],
                mode="lines",
                name=f"{model_name.title()}",
                line=dict(color=color, width=1.5, dash="dot"),
            ))

# Vertical line at forecast start
if not hist_data.empty:
    fig.add_vline(
        x=hist_data["date"].iloc[-1],
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
    )

fig.update_layout(
    title=f"{forecast_days}-Day Demand Forecast — {selected_cat}",
    xaxis_title="Date",
    yaxis_title="Daily Orders",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, use_container_width=True)


# ── Forecast Data Table ──
st.subheader("📋 Forecast Data")
st.dataframe(
    forecast_df[["date", "predicted_orders", "predicted_lower", "predicted_upper"]]
    .rename(columns={
        "date": "Date",
        "predicted_orders": "Predicted Orders",
        "predicted_lower": "Lower Bound (85%)",
        "predicted_upper": "Upper Bound (115%)",
    })
    .style.format({
        "Predicted Orders": "{:.0f}",
        "Lower Bound (85%)": "{:.0f}",
        "Upper Bound (115%)": "{:.0f}",
    }),
    hide_index=True,
    use_container_width=True,
)


# ── Download Forecast ──
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv,
    file_name=f"forecast_{selected_cat}_{forecast_days}days.csv",
    mime="text/csv",
)


# ── Model Info ──
st.subheader("ℹ️ Forecast Information")
st.json(info)
