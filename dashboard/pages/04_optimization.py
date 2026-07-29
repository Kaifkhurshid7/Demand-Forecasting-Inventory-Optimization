"""
Inventory Optimization Page — Professional SaaS-style PuLP optimizer interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR
from src.utils import load_model

st.set_page_config(layout="wide", page_title="Inventory Optimizer")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("Inventory Optimizer")
st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-top:-0.25rem;'>Optimize reorder quantities using linear programming</p>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="saas-card">
    <p style="margin:0;">
    This module uses <strong>Linear Programming (PuLP)</strong> to determine optimal reorder quantities
    per product category. The model minimizes <strong>holding costs + stockout costs</strong> while
    respecting <strong>storage capacity</strong> and <strong>budget</strong> constraints.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Check for forecast data ──
@st.cache_data
def get_top_categories_forecast():
    model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
    cats = sorted(set(f.stem.replace("ensemble_", "") for f in model_files))
    if not cats or daily.empty:
        return pd.DataFrame()

    recent = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=30)]
    top_cats = (
        recent.groupby(CATEGORY_COL)[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )

    results = []
    for cat in top_cats:
        cat_data = recent[recent[CATEGORY_COL] == cat]
        avg_demand = cat_data[TARGET_COL].mean()
        demand_std = cat_data[TARGET_COL].std()
        results.append({
            "category": cat,
            "avg_daily_demand": round(avg_demand, 1),
            "demand_std": round(demand_std, 1),
            "monthly_forecast": round(avg_demand * 30),
        })
    return pd.DataFrame(results)


forecast_data = get_top_categories_forecast()

if forecast_data.empty:
    st.warning("No forecast data available. Train models and run data pipeline first.")
    st.stop()

# ── Parameters Section ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Optimization Parameters</h2>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Service & Cost", "Capacity & Budget", "Lead Time"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        service_level = st.slider(
            "Service Level Target",
            min_value=0.80, max_value=0.99, value=0.95, step=0.01,
            help="Probability of not stocking out during lead time",
        )
    with col2:
        holding_cost_pct = st.number_input(
            "Annual Holding Cost (%)",
            min_value=5, max_value=50, value=25, step=5,
            help="Annual cost of holding inventory as % of unit cost",
        ) / 100

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        storage_capacity = st.number_input(
            "Storage Capacity (units)",
            min_value=1000, max_value=100000, value=10000, step=500,
            help="Total warehouse capacity in storage units",
        )
    with col2:
        budget = st.number_input(
            "Purchase Budget ($)",
            min_value=50000, max_value=5000000, value=500000, step=25000,
            help="Total budget for purchasing inventory",
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        lead_time = st.number_input(
            "Supplier Lead Time (days)",
            min_value=1, max_value=60, value=7, step=1,
        )
    with col2:
        stockout_cost_pct = st.number_input(
            "Stockout Cost (%)",
            min_value=10, max_value=100, value=40, step=5,
            help="Lost sale cost as % of unit cost",
        ) / 100

st.markdown("<br>", unsafe_allow_html=True)

# ── Cost Configuration ──
st.markdown("""
<div class="section-header">
    <div class="accent-bar"></div>
    <h2>Per-Category Cost Configuration</h2>
</div>
""", unsafe_allow_html=True)

cost_config = forecast_data[["category"]].copy()
if "avg_price" in daily.columns:
    avg_prices = (
        daily.groupby(CATEGORY_COL)["avg_price"]
        .mean()
        .reset_index()
        .rename(columns={"avg_price": "default_cost"})
    )
    cost_config = cost_config.merge(avg_prices, left_on="category", right_on=CATEGORY_COL, how="left")

cost_config["unit_cost"] = cost_config.get("default_cost", 50).fillna(50).round(2)
cost_config["selling_price"] = (cost_config["unit_cost"] * 1.6).round(2)
cost_config["current_stock"] = 100
cost_config["storage_per_unit"] = 1.0

edited_config = st.data_editor(
    cost_config[["category", "unit_cost", "selling_price", "current_stock", "storage_per_unit"]],
    column_config={
        "category": st.column_config.TextColumn("Category", disabled=True),
        "unit_cost": st.column_config.NumberColumn("Unit Cost ($)", min_value=1, max_value=10000, step=5),
        "selling_price": st.column_config.NumberColumn("Selling Price ($)", min_value=1, max_value=20000, step=5),
        "current_stock": st.column_config.NumberColumn("Current Stock", min_value=0, max_value=100000, step=10),
        "storage_per_unit": st.column_config.NumberColumn("Storage / Unit", min_value=0.1, max_value=100.0, step=0.5),
    },
    hide_index=True,
    use_container_width=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Run Optimization ──
run_col1, run_col2 = st.columns([1, 5])
with run_col1:
    run_clicked = st.button("Run Optimization", type="primary", use_container_width=True)

if run_clicked:
    from src.optimize import InventoryInput, solve_inventory_optimization

    items = []
    for _, row in edited_config.iterrows():
        cat = row["category"]
        cat_forecast = forecast_data[forecast_data["category"] == cat]
        demand = cat_forecast["monthly_forecast"].values[0] if len(cat_forecast) > 0 else 100
        demand_std = cat_forecast["demand_std"].values[0] if len(cat_forecast) > 0 else demand * 0.3

        items.append(InventoryInput(
            category=str(cat),
            forecast_demand=float(demand),
            forecast_std=float(max(demand_std, 1)),
            current_stock=float(row["current_stock"]),
            unit_cost=float(row["unit_cost"]),
            selling_price=float(row["selling_price"]),
            lead_time_days=lead_time,
            storage_per_unit=float(row["storage_per_unit"]),
        ))

    with st.spinner("Solving optimization problem..."):
        result = solve_inventory_optimization(
            items,
            storage_capacity=storage_capacity,
            budget=budget,
            service_level=service_level,
            holding_cost_pct=holding_cost_pct,
            stockout_cost_pct=stockout_cost_pct,
        )

    # ── Results ──
    st.markdown("""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>Optimization Results</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_color = "green" if result["status"] == "Optimal" else "red"
        st.metric("Status", result["status"])
    with col2:
        st.metric("Total Cost", f"${result['total_cost']:,.2f}")
    with col3:
        total_reorder = result["results"]["reorder_quantity"].sum()
        st.metric("Total Reorder Qty", f"{total_reorder:,.0f}")
    with col4:
        total_inv = result["results"]["total_inventory_after_order"].sum()
        st.metric("Total Inventory (Post)", f"{total_inv:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    results_df = result["results"].copy()
    display_results = results_df.rename(columns={
        "category": "Category",
        "current_stock": "Current Stock",
        "forecast_demand": "Forecast Demand",
        "reorder_quantity": "Reorder Qty",
        "reorder_point": "Reorder Point",
        "safety_stock": "Safety Stock",
        "total_inventory_after_order": "Total After Order",
        "unit_cost": "Unit Cost",
        "holding_cost": "Holding Cost",
    })

    st.dataframe(
        display_results.style.format({
            "Current Stock": "{:.0f}",
            "Forecast Demand": "{:.0f}",
            "Reorder Qty": "{:.0f}",
            "Reorder Point": "{:.0f}",
            "Safety Stock": "{:.0f}",
            "Total After Order": "{:.0f}",
            "Unit Cost": "${:.2f}",
            "Holding Cost": "${:.2f}",
        }).background_gradient(subset=["Reorder Qty"], cmap="Blues"),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualizations ──
    viz_tab1, viz_tab2 = st.tabs(["Reorder Quantities", "Cost Breakdown"])

    with viz_tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=results_df["category"],
            x=results_df["reorder_quantity"],
            orientation="h",
            name="Reorder Qty",
            marker_color="#3B82F6",
        ))
        fig.add_trace(go.Bar(
            y=results_df["category"],
            x=results_df["current_stock"],
            orientation="h",
            name="Current Stock",
            marker_color="#CBD5E1",
        ))
        fig.update_layout(
            title="Current Stock vs Reorder Quantity",
            xaxis_title="Units",
            yaxis_title="",
            barmode="group",
            template="plotly_white",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

    with viz_tab2:
        cost_fig = px.bar(
            results_df,
            x="category",
            y="holding_cost",
            title="Holding Cost by Category",
            labels={"holding_cost": "Holding Cost ($)", "category": ""},
            color="holding_cost",
            color_continuous_scale="Reds",
            template="plotly_white",
        )
        cost_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            coloraxis_showscale=False,
        )
        cost_fig.update_traces(marker_line_width=0)
        st.plotly_chart(cost_fig, use_container_width=True)

    # ── Download ──
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="inventory_optimization_results.csv",
        mime="text/csv",
    )

# ── Methodology ──
with st.expander("How It Works — Optimization Methodology"):
    st.markdown("""
    ### Inventory Optimization Model

    **Decision Variables:** Reorder quantity Q_i for each product category i.

    **Objective Function:** Minimize total inventory cost

    ```
    min SUM_i [ h * c_i * (I_i/2 + Q_i) + s * c_i * max(0, D_i - (I_i + Q_i)) ]
    ```

    Where:
    - **h** = holding cost rate (annual, as fraction of unit cost)
    - **c_i** = unit cost of category i
    - **I_i** = current inventory of category i
    - **Q_i** = reorder quantity for category i
    - **s** = stockout cost rate
    - **D_i** = forecast demand for category i

    ### Constraints

    | Constraint | Formula | Description |
    |---|---|---|
    | Storage Capacity | SUM_i w_i Q_i <= W | Total warehouse space |
    | Budget | SUM_i c_i Q_i <= B | Total purchase cost |
    | Service Level | I_i + Q_i >= D_i + z * sigma_i * sqrt(L/30) | Safety stock at target |
    | Non-negativity | Q_i >= 0 | No negative orders |

    The safety factor **z** is derived from the service level target using the inverse normal CDF
    (e.g., z = 1.645 for 95% service level, z = 2.326 for 99%).
    """)
