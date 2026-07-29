"""
Forecast Viewer Page — Professional SaaS-style demand forecast visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR
from src.utils import load_model
from src.predict import predict as generate_forecast

st.set_page_config(layout="wide", page_title="Forecast Viewer")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("Forecast Viewer")
st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-top:-0.25rem;'>Select a category and generate demand forecasts</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Discover available categories ──
model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
categories_available = sorted(set(
    f.stem.replace("ensemble_", "") for f in model_files
))

if not categories_available:
    st.warning("No trained models found. Please run `python src/train.py` first.")
    st.stop()

st.markdown(
    f"<div style='background:#ECFDF5; color:#065F46; padding:0.6rem 1rem; border-radius:8px; "
    f"font-size:0.85rem; margin-bottom:1.5rem;'>"
    f"[OK] {len(categories_available)} categories available for forecasting</div>",
    unsafe_allow_html=True
)

# ── Controls ──
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_cat = st.selectbox(
        "Product Category",
        categories_available,
        index=categories_available.index("bed_bath_table") if "bed_bath_table" in categories_available else 0,
    )
with col2:
    forecast_days = st.selectbox(
        "Forecast Horizon",
        [7, 14, 30, 60, 90],
        index=2,
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
    st.metric("Avg Daily (Forecast)", f"{info['avg_daily_forecast']:.1f}")
with col3:
    st.metric("Avg Daily (History)", f"{info['avg_historical_daily']:.1f}")
with col4:
    change = ((info['avg_daily_forecast'] - info['avg_historical_daily'])
              / max(info['avg_historical_daily'], 0.01) * 100)
    direction = "+" if change >= 0 else ""
    st.metric("Trend vs History", f"{direction}{change:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Forecast Plot ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Forecast Visualization</h2>
</div>
""", unsafe_allow_html=True)

# Get historical data
if not daily.empty:
    hist_data = daily[daily[CATEGORY_COL] == selected_cat].sort_values("date").tail(90)
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
        line=dict(color="#94A3B8", width=1.5),
    ))

# Forecast with confidence band
last_date = hist_data["date"].iloc[-1] if not hist_data.empty else forecast_df["date"].iloc[0]
last_val = hist_data[TARGET_COL].iloc[-1] if not hist_data.empty else None

# Upper bound
fig.add_trace(go.Scatter(
    x=pd.concat([
        pd.Series([last_date]) if not hist_data.empty else pd.Series([]),
        forecast_df["date"]
    ]),
    y=pd.concat([
        pd.Series([last_val]) if last_val is not None else pd.Series([]),
        forecast_df["predicted_upper"]
    ]),
    mode="lines",
    name="Upper Bound",
    line=dict(width=0),
    showlegend=False,
    fill=None,
))
# Lower bound with fill
fig.add_trace(go.Scatter(
    x=pd.concat([
        pd.Series([last_date]) if not hist_data.empty else pd.Series([]),
        forecast_df["date"]
    ]),
    y=pd.concat([
        pd.Series([last_val]) if last_val is not None else pd.Series([]),
        forecast_df["predicted_lower"]
    ]),
    mode="lines",
    name="Confidence Band",
    line=dict(width=0),
    fillcolor="rgba(59, 130, 246, 0.15)",
    fill="tonexty",
    showlegend=True,
))

# Mean forecast
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["predicted_orders"],
    mode="lines+markers",
    name="Forecast (Ensemble)",
    line=dict(color="#1E3A5F", width=2.5),
    marker=dict(size=5, color="#1E3A5F", symbol="circle"),
))

# Individual model predictions
if show_individual:
    model_styles = {
        "xgboost": {"color": "#10B981", "dash": "dot", "label": "XGBoost"},
        "prophet": {"color": "#F59E0B", "dash": "dot", "label": "Prophet"},
        "lstm": {"color": "#8B5CF6", "dash": "dot", "label": "LSTM"},
    }
    for model_name, style in model_styles.items():
        col_name = f"{model_name}_prediction"
        if col_name in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[col_name],
                mode="lines",
                name=style["label"],
                line=dict(color=style["color"], width=1.5, dash=style["dash"]),
            ))

# Forecast start line
if not hist_data.empty:
    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color="#64748B",
        annotation_text="Forecast Start",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#64748B",
    )

fig.update_layout(
    title=f"{forecast_days}-Day Demand Forecast — {selected_cat}",
    xaxis_title=None,
    yaxis_title="Daily Orders",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12, font_size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast Data Table ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Forecast Data</h2>
</div>
""", unsafe_allow_html=True)

display_fc = forecast_df[["date", "predicted_orders", "predicted_lower", "predicted_upper"]].copy()
display_fc.columns = ["Date", "Predicted Orders", "Lower Bound (85%)", "Upper Bound (115%)"]

st.dataframe(
    display_fc.style.format({
        "Predicted Orders": "{:.0f}",
        "Lower Bound (85%)": "{:.0f}",
        "Upper Bound (115%)": "{:.0f}",
    }),
    hide_index=True,
    use_container_width=True,
)

# ── Download ──
col1, col2 = st.columns([1, 4])
with col1:
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"forecast_{selected_cat}_{forecast_days}days.csv",
        mime="text/csv",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Info ──
with st.expander("Forecast Information"):
    clean_info = {k: v for k, v in info.items()}
    st.json(clean_info)
