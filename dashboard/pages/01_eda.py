"""
EDA Overview Page — Explore sales trends, seasonality, and category patterns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.config import CATEGORY_COL, TARGET_COL

st.set_page_config(layout="wide", page_title="EDA Overview")

daily = st.session_state.get("daily_data", pd.DataFrame())

if daily.empty:
    st.warning("No data loaded. Please run `python src/data_loader.py` first.")
    st.stop()

st.title("📈 Exploratory Data Analysis")
st.markdown("---")

# ── Summary Metrics ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Orders", f"{int(daily[TARGET_COL].sum()):,}")
with col2:
    avg_daily = daily.groupby("date")[TARGET_COL].sum().mean()
    st.metric("Avg Daily Orders", f"{avg_daily:.0f}")
with col3:
    st.metric("Categories", daily[CATEGORY_COL].nunique())
with col4:
    date_range = f"{daily['date'].min().date()} → {daily['date'].max().date()}"
    st.metric("Date Range", date_range)


# ── Daily Order Trend ──
st.subheader("📆 Daily Order Trend")
daily_total = daily.groupby("date")[TARGET_COL].sum().reset_index()

fig = px.line(
    daily_total,
    x="date",
    y=TARGET_COL,
    title="Total Daily Orders (All Categories)",
    labels={"order_count": "Orders", "date": "Date"},
    template="plotly_white",
)
fig.add_scatter(
    x=daily_total["date"],
    y=daily_total[TARGET_COL].rolling(14, min_periods=1).mean(),
    mode="lines",
    line=dict(color="red", width=2),
    name="14-day MA",
)
st.plotly_chart(fig, use_container_width=True)


# ── Top Categories ──
st.subheader("🏆 Top Categories by Volume")
top_n = st.slider("Number of categories to show", 5, 30, 15)

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
    title=f"Top {top_n} Categories by Total Orders",
    labels={"order_count": "Total Orders", "product_category_name_english": ""},
    color=TARGET_COL,
    color_continuous_scale="Blues",
    template="plotly_white",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)


# ── Category Selector for Detailed View ──
st.subheader("🔍 Category Deep Dive")
categories = sorted(daily[CATEGORY_COL].unique().tolist())
selected_cat = st.selectbox("Select a category to explore", categories)

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
        line=dict(color="red", width=2),
        name="7-day MA",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Day-of-week pattern
    cat_data_copy = cat_data.copy()
    cat_data_copy["day_of_week"] = pd.to_datetime(cat_data_copy["date"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = cat_data_copy.groupby("day_of_week")[TARGET_COL].mean().reindex(dow_order).reset_index()

    fig = px.bar(
        dow_avg,
        x="day_of_week",
        y=TARGET_COL,
        title=f"Average Orders by Day of Week — {selected_cat}",
        labels={"order_count": "Avg Orders", "day_of_week": ""},
        color=TARGET_COL,
        color_continuous_scale="Viridis",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Seasonality Heatmap ──
st.subheader("📅 Seasonality Heatmap")
st.markdown("Average daily orders by month and day of week.")

cat_data_hm = cat_data.copy()
cat_data_hm["month"] = pd.to_datetime(cat_data_hm["date"]).dt.month
cat_data_hm["dow"] = pd.to_datetime(cat_data_hm["date"]).dt.dayofweek

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

heatmap_data = cat_data_hm.pivot_table(
    values=TARGET_COL, index="dow", columns="month", aggfunc="mean"
)

fig = px.imshow(
    heatmap_data.values,
    x=[month_names[m - 1] for m in heatmap_data.columns],
    y=[dow_names[d] for d in heatmap_data.index],
    color_continuous_scale="YlOrRd",
    title=f"Average Orders — {selected_cat}",
    labels={"x": "Month", "y": "Day of Week", "color": "Avg Orders"},
    template="plotly_white",
    aspect="auto",
)
st.plotly_chart(fig, use_container_width=True)


# ── Revenue Analysis ──
if "total_revenue" in daily.columns:
    st.subheader("💰 Revenue Analysis")
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
        title=f"Top {top_n} Categories by Total Revenue",
        labels={"total_revenue": "Revenue (R$)", "product_category_name_english": ""},
        color="total_revenue",
        color_continuous_scale="Greens",
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ── Category Growth Rate ──
st.subheader("📈 Category Growth Trend (Month-over-Month)")
cat_data_mom = cat_data.copy()
cat_data_mom["month"] = pd.to_datetime(cat_data_mom["date"]).dt.to_period("M").astype(str)
monthly = cat_data_mom.groupby("month")[TARGET_COL].sum().reset_index()

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
st.plotly_chart(fig, use_container_width=True)
