"""
Model Comparison Page — Professional SaaS-style model performance comparison.
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
st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-top:-0.25rem;'>Forecast performance across models and categories</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Load Metrics ──
@st.cache_data
def load_saved_metrics():
    metrics_file = MODELS_DIR / "all_categories_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {}

saved_metrics = load_saved_metrics()

# ── Check model availability ──
model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
categories_available = sorted(set(
    f.stem.replace("ensemble_", "") for f in model_files
))

if not categories_available:
    st.warning("No trained models found. Please run `python src/train.py` first.")
    st.info("Run `python src/train.py` to train ensemble models for each category.")
    st.stop()

st.markdown(
    f"<div style='background:#ECFDF5; color:#065F46; padding:0.6rem 1rem; border-radius:8px; "
    f"font-size:0.85rem; margin-bottom:1.5rem;'>"
    f"[OK] Models loaded for {len(categories_available)} categories</div>",
    unsafe_allow_html=True
)

# ── Aggregate Metrics ──
if saved_metrics:
    st.markdown("""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>Aggregate Performance Summary</h2>
    </div>
    """, unsafe_allow_html=True)

    metrics_df = pd.DataFrame.from_dict(saved_metrics, orient="index")
    metrics_df.index.name = "category"
    metrics_df = metrics_df.reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_mape = metrics_df["mape"].mean()
        st.metric("Average MAPE", f"{avg_mape:.2f}%",
                  delta=f"{'Below' if avg_mape < 15 else 'Above'} 15% target",
                  delta_color="inverse" if avg_mape < 15 else "off")
    with col2:
        st.metric("Average RMSE", f"{metrics_df['rmse'].mean():.2f}")
    with col3:
        st.metric("Average MAE", f"{metrics_df['mae'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # MAPE Distribution
    fig = px.histogram(
        metrics_df,
        x="mape",
        nbins=20,
        title="MAPE Distribution Across Categories",
        labels={"mape": "MAPE (%)"},
        template="plotly_white",
        color_discrete_sequence=["#3B82F6"],
    )
    fig.add_vline(x=15, line_dash="dash", line_color="#EF4444",
                  annotation_text="Target (<15%)")
    fig.update_layout(
        hovermode="x",
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>Category-Level Metrics</h2>
    </div>
    """, unsafe_allow_html=True)

    display_df = metrics_df.sort_values("mape").rename(columns={
        "category": "Category", "mape": "MAPE (%)", "rmse": "RMSE", "mae": "MAE"
    }).reset_index(drop=True)

    # Highlight rows where MAPE < 15%
    def highlight_row(row):
        if row["MAPE (%)"] < 15:
            return ["background: #ECFDF5"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style
        .format({"MAPE (%)": "{:.2f}%", "RMSE": "{:.2f}", "MAE": "{:.2f}"})
        .apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

# ── Per-Category Forecast vs Actual ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Forecast vs Actual</h2>
</div>
""", unsafe_allow_html=True)

selected_cat = st.selectbox("Select category to examine", categories_available, label_visibility="collapsed")

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
            line=dict(color="#0F172A", width=2),
            marker=dict(size=4, color="#0F172A"),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=baseline,
            mode="lines",
            name="7-day MA Baseline",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
        ))

        fig.update_layout(
            title=f"Recent Demand — {selected_cat}",
            xaxis_title="Date",
            yaxis_title="Daily Orders",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.12),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(actual[7:], baseline[7:])
        st.markdown(
            f"<span class='badge badge-blue'>Baseline MAE: {mae:.2f}</span>",
            unsafe_allow_html=True
        )

    # Category statistics
    st.markdown("""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>Category Statistics</h2>
    </div>
    """, unsafe_allow_html=True)

    stats = cat_data[TARGET_COL].describe()
    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "25th Percentile", "Median", "75th Percentile", "Max"],
        "Value": [
            f"{stats['mean']:.1f}", f"{stats['std']:.1f}",
            f"{stats['min']:.0f}", f"{stats['25%']:.0f}",
            f"{stats['50%']:.0f}", f"{stats['75%']:.0f}",
            f"{stats['max']:.0f}",
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

# ── Model Architecture Summary ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Model Architecture</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="saas-card" style="height:100%;">
        <h3>Prophet</h3>
        <p class="subtext">Additive model</p>
        <hr style="margin:0.5rem 0;">
        <p style="font-size:0.82rem;">
        Yearly &amp; weekly seasonality. Native holiday regressors.
        </p>
        <p style="font-size:0.82rem; font-weight:500; margin-top:0.5rem;">
        Best for: Trend + strong seasonality
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="saas-card" style="height:100%;">
        <h3>XGBoost</h3>
        <p class="subtext">Gradient boosting</p>
        <hr style="margin:0.5rem 0;">
        <p style="font-size:0.82rem;">
        Lags, rolling stats, seasonal dummies. Handles tabular features.
        </p>
        <p style="font-size:0.82rem; font-weight:500; margin-top:0.5rem;">
        Best for: Feature-rich demand patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="saas-card" style="height:100%;">
        <h3>LSTM</h3>
        <p class="subtext">Recurrent neural network</p>
        <hr style="margin:0.5rem 0;">
        <p style="font-size:0.82rem;">
        2-layer network. 30-day sliding window sequences.
        </p>
        <p style="font-size:0.82rem; font-weight:500; margin-top:0.5rem;">
        Best for: Long-range temporal dependencies
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="saas-card" style="height:100%;">
        <h3>Ensemble</h3>
        <p class="subtext">Weighted average</p>
        <hr style="margin:0.5rem 0;">
        <p style="font-size:0.82rem;">
        Combines all models. Weights via validation RMSE.
        </p>
        <p style="font-size:0.82rem; font-weight:500; margin-top:0.5rem;">
        Best for: Robust, stable forecasts
        </p>
    </div>
    """, unsafe_allow_html=True)
