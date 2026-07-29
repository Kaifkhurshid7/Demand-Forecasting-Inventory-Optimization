"""
Model Comparison Page — Dark theme model performance comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR

st.set_page_config(layout="wide", page_title="Model Comparison")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("Model Comparison")
st.markdown("<p style='color:#8B949E; font-size:0.85rem; margin-top:-0.25rem;'>Forecast performance across models and categories</p>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_saved_metrics():
    metrics_file = MODELS_DIR / "all_categories_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {}

saved_metrics = load_saved_metrics()

model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
categories_available = sorted(set(f.stem.replace("ensemble_", "") for f in model_files))

if not categories_available:
    st.warning("No trained models found. Run `python src/train.py` first.")
    st.stop()

st.markdown(
    f"<div style='background:#0C2D1B; color:#3FB950; padding:0.5rem 1rem; border-radius:6px; "
    f"font-size:0.82rem; margin-bottom:1.25rem; border:1px solid #1A3F2B;'>"
    f"OK  Models loaded for {len(categories_available)} categories</div>",
    unsafe_allow_html=True
)

if saved_metrics:
    st.markdown("""
    <div class="section-header">
        <div class="bar"></div>
        <h2>Aggregate Performance</h2>
    </div>
    """, unsafe_allow_html=True)

    metrics_df = pd.DataFrame.from_dict(saved_metrics, orient="index")
    metrics_df.index.name = "category"
    metrics_df = metrics_df.reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_mape = metrics_df["mape"].mean()
        st.metric("Average MAPE", f"{avg_mape:.2f}%",
                  delta="Below 15% target" if avg_mape < 15 else "Above 15% target",
                  delta_color="inverse" if avg_mape < 15 else "off")
    with col2:
        st.metric("Average RMSE", f"{metrics_df['rmse'].mean():.2f}")
    with col3:
        st.metric("Average MAE", f"{metrics_df['mae'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = px.histogram(
        metrics_df,
        x="mape",
        nbins=20,
        title="",
        labels={"mape": "MAPE (%)"},
        template="plotly_dark",
        color_discrete_sequence=["#58A6FF"],
    )
    fig.add_vline(x=15, line_dash="dash", line_color="#F85149",
                  annotation_text="Target <15%")
    fig.update_layout(
        hovermode="x",
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#0D1117",
        font=dict(color="#C9D1D9"),
        xaxis=dict(gridcolor="#21262D"),
        yaxis=dict(gridcolor="#21262D"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="section-header">
        <div class="bar"></div>
        <h2>Category-Level Metrics</h2>
    </div>
    """, unsafe_allow_html=True)

    display_df = metrics_df.sort_values("mape").rename(columns={
        "category": "Category", "mape": "MAPE (%)", "rmse": "RMSE", "mae": "MAE"
    }).reset_index(drop=True)

    def highlight_row(row):
        if row["MAPE (%)"] < 15:
            return ["background: #0C2D1B; color: #3FB950"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.format({"MAPE (%)": "{:.2f}%", "RMSE": "{:.2f}", "MAE": "{:.2f}"}).apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("""
<div class="section-header">
    <div class="bar"></div>
    <h2>Forecast vs Actual</h2>
</div>
""", unsafe_allow_html=True)

selected_cat = st.selectbox("Select category", categories_available, label_visibility="collapsed")

if selected_cat and not featured.empty:
    cat_data = featured[featured[CATEGORY_COL] == selected_cat].sort_values("date")
    test_data = cat_data.tail(60).copy()

    if len(test_data) > 0:
        actual = test_data[TARGET_COL].values
        dates = test_data["date"].values
        baseline = test_data[TARGET_COL].shift(1).rolling(7, min_periods=1).mean().values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#F0F6FC", width=2),
            marker=dict(size=4, color="#F0F6FC"),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=baseline,
            mode="lines",
            name="7-day MA Baseline",
            line=dict(color="#8B949E", width=1.5, dash="dash"),
        ))

        fig.update_layout(
            title=f"Recent Demand - {selected_cat}",
            xaxis_title="Date",
            yaxis_title="Daily Orders",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.12, font=dict(color="#8B949E")),
            plot_bgcolor="#0D1117",
            paper_bgcolor="#0D1117",
            font=dict(color="#C9D1D9"),
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D"),
        )
        st.plotly_chart(fig, use_container_width=True)

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(actual[7:], baseline[7:])
        st.markdown(f"<span class='badge badge-blue'>Baseline MAE: {mae:.2f}</span>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="bar"></div>
        <h2>Category Statistics</h2>
    </div>
    """, unsafe_allow_html=True)

    stats = cat_data[TARGET_COL].describe()
    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
        "Value": [
            f"{stats['mean']:.1f}", f"{stats['std']:.1f}",
            f"{stats['min']:.0f}", f"{stats['25%']:.0f}",
            f"{stats['50%']:.0f}", f"{stats['75%']:.0f}",
            f"{stats['max']:.0f}",
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

st.markdown("""
<div class="section-header">
    <div class="bar"></div>
    <h2>Model Architecture</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card" style="height:100%;">
        <h3>Prophet</h3>
        <span class="tag">Additive model</span>
        <hr style="margin:0.5rem 0;">
        <p>Yearly and weekly seasonality. Native holiday regressors.</p>
        <p style="margin-top:0.5rem; color:#58A6FF;">Best for: Trend + seasonality</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="height:100%;">
        <h3>XGBoost</h3>
        <span class="tag">Gradient boosting</span>
        <hr style="margin:0.5rem 0;">
        <p>Lags, rolling stats, seasonal dummies. Strong with tabular features.</p>
        <p style="margin-top:0.5rem; color:#58A6FF;">Best for: Feature-rich patterns</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card" style="height:100%;">
        <h3>LSTM</h3>
        <span class="tag">Recurrent network</span>
        <hr style="margin:0.5rem 0;">
        <p>2-layer neural network. 30-day sliding window sequences.</p>
        <p style="margin-top:0.5rem; color:#58A6FF;">Best for: Temporal dependencies</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card" style="height:100%;">
        <h3>Ensemble</h3>
        <span class="tag">Weighted average</span>
        <hr style="margin:0.5rem 0;">
        <p>Combines all models. Weights from validation RMSE.</p>
        <p style="margin-top:0.5rem; color:#58A6FF;">Best for: Robust forecasts</p>
    </div>
    """, unsafe_allow_html=True)
