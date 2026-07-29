"""
Inventory Optimization Page — Run the PuLP solver and view reorder recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.config import CATEGORY_COL, TARGET_COL, MODELS_DIR, INVENTORY_PARAMS
from src.utils import load_model

st.set_page_config(layout="wide", page_title="Inventory Optimizer")

daily = st.session_state.get("daily_data", pd.DataFrame())
featured = st.session_state.get("featured_data", pd.DataFrame())

st.title("[$] Inventory Optimizer")
st.markdown("---")

st.markdown("""
This module uses **Linear Programming (PuLP)** to determine optimal reorder quantities
per product category. The model minimizes **holding costs + stockout costs** while
respecting **storage capacity** and **budget** constraints.
""")


# -- Check for forecast data --
@st.cache_data
def get_top_categories_forecast():
    """Get forecast data for top categories if models exist."""
    model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
    cats = sorted(set(f.stem.replace("ensemble_", "") for f in model_files))

    if not cats or daily.empty:
        return pd.DataFrame()

    # Get latest daily data for top categories
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

# -- Parameters --
st.subheader("[=] Optimization Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    service_level = st.slider(
        "Service Level",
        min_value=0.80, max_value=0.99, value=0.95, step=0.01,
        help="Probability of not stocking out during lead time",
    )
    holding_cost_pct = st.number_input(
        "Annual Holding Cost (%)",
        min_value=5, max_value=50, value=25, step=5,
        help="Annual cost of holding inventory as % of unit cost",
    ) / 100

with col2:
    storage_capacity = st.number_input(
        "Storage Capacity (units)",
        min_value=1000, max_value=100000, value=10000, step=500,
        help="Total warehouse capacity in storage units",
    )
    stockout_cost_pct = st.number_input(
        "Stockout Cost (%)",
        min_value=10, max_value=100, value=40, step=5,
        help="Lost sale cost as % of unit cost",
    ) / 100

with col3:
    budget = st.number_input(
        "Purchase Budget ($)",
        min_value=50000, max_value=5000000, value=500000, step=25000,
        help="Total budget for purchasing inventory",
    )
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1, max_value=60, value=7, step=1,
        help="Supplier delivery lead time",
    )


# -- Cost Configuration --
st.subheader("[$] Per-Category Cost Configuration")

# Let user edit unit costs
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
cost_config["current_stock"] = 100  # default
cost_config["storage_per_unit"] = 1.0

# Editable table
edited_config = st.data_editor(
    cost_config[["category", "unit_cost", "selling_price", "current_stock", "storage_per_unit"]],
    column_config={
        "category": st.column_config.TextColumn("Category", disabled=True),
        "unit_cost": st.column_config.NumberColumn("Unit Cost ($)", min_value=1, max_value=10000, step=5),
        "selling_price": st.column_config.NumberColumn("Selling Price ($)", min_value=1, max_value=20000, step=5),
        "current_stock": st.column_config.NumberColumn("Current Stock", min_value=0, max_value=100000, step=10),
        "storage_per_unit": st.column_config.NumberColumn("Storage per Unit", min_value=0.1, max_value=100.0, step=0.5),
    },
    hide_index=True,
    use_container_width=True,
)


# -- Run Optimization --
if st.button("[>] Run Optimization", type="primary"):
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

    st.subheader("[OK] Optimization Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", result["status"])
    with col2:
        st.metric("Total Cost", f"${result['total_cost']:,.2f}")
    with col3:
        total_reorder = result["results"]["reorder_quantity"].sum()
        st.metric("Total Reorder Qty", f"{total_reorder:,.0f}")

    # Results table
    results_df = result["results"].copy()
    st.dataframe(
        results_df.style
        .format({
            "current_stock": "{:.0f}",
            "forecast_demand": "{:.0f}",
            "reorder_quantity": "{:.0f}",
            "reorder_point": "{:.0f}",
            "safety_stock": "{:.0f}",
            "total_inventory_after_order": "{:.0f}",
            "unit_cost": "${:.2f}",
            "holding_cost": "${:.2f}",
        })
        .background_gradient(subset=["reorder_quantity"], cmap="Blues"),
        hide_index=True,
        use_container_width=True,
    )

    # -- Visualization --
    st.subheader("[#] Reorder Recommendations")

    # Horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=results_df["category"],
        x=results_df["reorder_quantity"],
        orientation="h",
        name="Reorder Qty",
        marker_color="#2E86AB",
    ))
    fig.add_trace(go.Bar(
        y=results_df["category"],
        x=results_df["current_stock"],
        orientation="h",
        name="Current Stock",
        marker_color="#A0A0A0",
    ))

    fig.update_layout(
        title="Current Stock vs Reorder Quantity by Category",
        xaxis_title="Units",
        yaxis_title="",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    st.subheader("[$] Cost Breakdown")

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
    st.plotly_chart(cost_fig, use_container_width=True)

    # Download results
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="[V] Download Optimization Results",
        data=csv,
        file_name="inventory_optimization_results.csv",
        mime="text/csv",
    )


# -- Methodology Explanation --
with st.expander("[i] How It Works"):
    st.markdown("""
    ### Inventory Optimization Model

    **Decision Variables:** Reorder quantity Q_i for each product category i.

    **Objective Function:** Minimize total inventory cost

    ```
    min SUM_i [ h * c_i * (I_i/2 + Q_i) + s * c_i * max(0, D_i - (I_i + Q_i)) ]
    ```

    Where:
    - h = holding cost rate (annual, as fraction of unit cost)
    - c_i = unit cost of category i
    - I_i = current inventory of category i
    - Q_i = reorder quantity for category i
    - s = stockout cost rate
    - D_i = forecast demand for category i

    **Constraints:**
    - **Storage capacity:** SUM_i w_i Q_i <= W (total space)
    - **Budget:** SUM_i c_i Q_i <= B (total purchase cost)
    - **Service level:** I_i + Q_i >= D_i + z * sigma_i * sqrt(L/30)
    - **Non-negativity:** Q_i >= 0

    The safety factor z is derived from the service level target
    (e.g., z = 1.645 for 95% service level).
    """)
