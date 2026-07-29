"""
EDA Overview Page — Professional SaaS-style exploratory data analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.config import CATEGORY_COL, TARGET_COL

st.set_page_config(layout="wide", page_title="EDA Overview")

daily = st.session_state.get("daily_data", pd.DataFrame())

if daily.empty:
    st.warning("No data loaded. Please run `python src/data_loader.py` first.")
    st.stop()

st.title("Exploratory Data Analysis")
st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-top:-0.25rem;'>Sales trends, seasonality, and category patterns</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Summary Metrics ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    total = int(daily[TARGET_COL].sum())
    st.metric("Total Orders", f"{total:,}")
with col2:
    avg_daily = daily.groupby("date")[TARGET_COL].sum().mean()
    st.metric("Avg Daily Orders", f"{avg_daily:.0f}")
with col3:
    st.metric("Categories Tracked", daily[CATEGORY_COL].nunique())
with col4:
    dr = f"{daily['date'].min().date()} - {daily['date'].max().date()}"
    st.metric("Date Range", dr)

st.markdown("<br>", unsafe_allow_html=True)

# ── Daily Order Trend ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Daily Order Trend</h2>
</div>
""", unsafe_allow_html=True)

daily_total = daily.groupby("date")[TARGET_COL].sum().reset_index()

fig = px.line(
    daily_total,
    x="date",
    y=TARGET_COL,
    title="Total Daily Orders (All Categories)",
    labels={"order_count": "Orders", "date": ""},
    template="plotly_white",
)
fig.add_scatter(
    x=daily_total["date"],
    y=daily_total[TARGET_COL].rolling(14, min_periods=1).mean(),
    mode="lines",
    line=dict(color="#3B82F6", width=2.5),
    name="14-day Moving Avg",
)
fig.update_layout(
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
)
st.plotly_chart(fig, use_container_width=True)

# ── Top Categories ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Top Categories by Volume</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col2:
    top_n = st.selectbox("Show top", [10, 15, 20, 30], index=1)

cat_totals = (
    daily.groupby(CATEGORY_COL)[TARGET_COL]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .reset_index()
)

fig = px.bar(
    cat_totals,
    x=TARGET_COL,
    y=CATEGORY_COL,
    orientation="h",
    title=f"Top {top_n} Categories",
    labels={"order_count": "Total Orders", "product_category_name_english": ""},
    color=TARGET_COL,
    color_continuous_scale="Blues",
    template="plotly_white",
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    coloraxis_showscale=False,
)
fig.update_traces(marker_line_width=0)
st.plotly_chart(fig, use_container_width=True)

# ── Category Deep Dive ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Category Deep Dive</h2>
</div>
""", unsafe_allow_html=True)

categories = sorted(daily[CATEGORY_COL].unique().tolist())
selected_cat = st.selectbox("Select category to explore", categories, label_visibility="collapsed")

cat_data = daily[daily[CATEGORY_COL] == selected_cat].sort_values("date")

col1, col2 = st.columns(2)

with col1:
    fig = px.line(
        cat_data,
        x="date",
        y=TARGET_COL,
        title=f"Daily Orders — {selected_cat}",
        labels={"order_count": "Orders", "date": ""},
        template="plotly_white",
    )
    fig.add_scatter(
        x=cat_data["date"],
        y=cat_data[TARGET_COL].rolling(7, min_periods=1).mean(),
        mode="lines",
        line=dict(color="#3B82F6", width=2.5),
        name="7-day MA",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cat_copy = cat_data.copy()
    cat_copy["dow"] = pd.to_datetime(cat_copy["date"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = cat_copy.groupby("dow")[TARGET_COL].mean().reindex(dow_order).reset_index()

    fig = px.bar(
        dow_avg,
        x="dow",
        y=TARGET_COL,
        title=f"Avg Orders by Day — {selected_cat}",
        labels={"order_count": "Avg Orders", "dow": ""},
        color=TARGET_COL,
        color_continuous_scale="Viridis",
        template="plotly_white",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        coloraxis_showscale=False,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# ── Seasonality Heatmap ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Seasonality Heatmap</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("<p class='text-muted'>Average daily orders by month and day of week</p>", unsafe_allow_html=True)

cat_hm = cat_data.copy()
cat_hm["month"] = pd.to_datetime(cat_hm["date"]).dt.month
cat_hm["dow"] = pd.to_datetime(cat_hm["date"]).dt.dayofweek

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

heatmap_data = cat_hm.pivot_table(values=TARGET_COL, index="dow", columns="month", aggfunc="mean")

fig = px.imshow(
    heatmap_data.values,
    x=[month_names[m - 1] for m in heatmap_data.columns],
    y=[dow_names[d] for d in heatmap_data.index],
    color_continuous_scale="YlOrRd",
    title="",
    labels={"x": "Month", "y": "Day of Week", "color": "Avg"},
    template="plotly_white",
    aspect="auto",
)
fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
)
st.plotly_chart(fig, use_container_width=True)

# ── Revenue Analysis ──
if "total_revenue" in daily.columns:
    st.markdown("""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>Revenue Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    revenue_data = (
        daily.groupby(CATEGORY_COL)["total_revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        revenue_data,
        x="total_revenue",
        y=CATEGORY_COL,
        orientation="h",
        title=f"Top {top_n} Categories by Revenue",
        labels={"total_revenue": "Revenue (R$)", "product_category_name_english": ""},
        color="total_revenue",
        color_continuous_scale="Greens",
        template="plotly_white",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        coloraxis_showscale=False,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# ── Category Growth ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Monthly Growth Trend</h2>
</div>
""", unsafe_allow_html=True)

cat_mom = cat_data.copy()
cat_mom["month"] = pd.to_datetime(cat_mom["date"]).dt.to_period("M").astype(str)
monthly = cat_mom.groupby("month")[TARGET_COL].sum().reset_index()

fig = px.bar(
    monthly,
    x="month",
    y=TARGET_COL,
    title=f"Monthly Order Volume — {selected_cat}",
    labels={"order_count": "Orders", "month": ""},
    color=TARGET_COL,
    color_continuous_scale="Blues",
    template="plotly_white",
)
fig.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    coloraxis_showscale=False,
)
fig.update_traces(marker_line_width=0)
st.plotly_chart(fig, use_container_width=True)
