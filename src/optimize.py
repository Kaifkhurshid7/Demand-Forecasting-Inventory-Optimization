"""
Inventory Optimization — Linear Programming (PuLP) solver.

Determines optimal reorder quantities per product category to minimize
holding and stockout costs while respecting capacity and service constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.config import INVENTORY_PARAMS, MODELS_DIR, CATEGORY_COL, TARGET_COL

logger = logging.getLogger(__name__)


@dataclass
class InventoryInput:
    """Input data for the inventory optimization model."""
    category: str
    forecast_demand: float      # Expected demand for the period
    forecast_std: float         # Demand uncertainty (std dev)
    current_stock: float        # Units currently in stock
    unit_cost: float            # Purchase cost per unit
    selling_price: float        # Selling price per unit
    lead_time_days: int         # Supplier lead time
    storage_per_unit: float     # Storage space per unit (cubic ft / slots)


def solve_inventory_optimization(
    items: List[InventoryInput],
    storage_capacity: Optional[float] = None,
    budget: Optional[float] = None,
    service_level: Optional[float] = None,
    holding_cost_pct: Optional[float] = None,
    stockout_cost_pct: Optional[float] = None,
) -> Dict:
    """
    Solve the multi-item inventory optimization problem using linear programming.

    The model:
        Decision Variables: Reorder quantity (Q_i) for each category
        Objective: Minimize total cost = holding cost + stockout cost
        Constraints:
            - Storage capacity (total space ≤ capacity)
            - Budget (total purchase cost ≤ budget)
            - Non-negativity (Q_i ≥ 0)

    Args:
        items: List of InventoryInput for each category.
        storage_capacity: Total warehouse capacity in units of space.
        budget: Maximum purchase budget.
        service_level: Desired service level (0-1).
        holding_cost_pct: Annual holding cost as fraction of unit cost.
        stockout_cost_pct: Stockout cost as fraction of unit cost.

    Returns:
        Dict with:
            - 'results': DataFrame with reorder quantities and metrics
            - 'total_cost': Total expected inventory cost
            - 'status': Solution status
    """
    import pulp

    # Defaults
    storage_capacity = storage_capacity or INVENTORY_PARAMS["storage_capacity"]
    budget = budget or INVENTORY_PARAMS["budget"]
    service_level = service_level or INVENTORY_PARAMS["service_level"]
    holding_cost_pct = holding_cost_pct or INVENTORY_PARAMS["holding_cost_pct"]
    stockout_cost_pct = stockout_cost_pct or INVENTORY_PARAMS["stockout_cost_pct"]

    # Safety factor for service level (assumes normally distributed demand)
    from scipy.stats import norm
    safety_factor = norm.ppf(service_level)

    # Create the LP problem
    prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

    # Decision variables: reorder quantity per category
    n = len(items)
    Q = [pulp.LpVariable(f"Q_{i}", lowBound=0, cat="Continuous") for i in range(n)]

    # ── Objective: minimize total cost ──
    # Total cost = holding cost + stockout cost
    #   Holding cost  = holding_cost_pct * unit_cost * (avg inventory level)
    #   Stockout cost = stockout_cost_pct * unit_cost * expected stockout
    #   Avg inventory ≈ current_stock/2 + Q (simplified periodic review)

    total_cost = pulp.lpSum([
        (
            holding_cost_pct * item.unit_cost * (item.current_stock / 2.0 + Q[i])
            + stockout_cost_pct * item.unit_cost
            * max(0, item.forecast_demand - (item.current_stock + Q[i]))
        )
        for i, item in enumerate(items)
    ])
    prob += total_cost

    # ── Constraints ──

    # 1. Storage capacity
    prob += pulp.lpSum([item.storage_per_unit * Q[i] for i, item in enumerate(items)]) <= storage_capacity, "StorageCapacity"

    # 2. Budget
    prob += pulp.lpSum([item.unit_cost * Q[i] for i, item in enumerate(items)]) <= budget, "Budget"

    # 3. Safety stock (service level): Q_i + current_stock ≥ forecast_demand + safety_stock
    for i, item in enumerate(items):
        safety_stock = safety_factor * item.forecast_std * np.sqrt(item.lead_time_days / 30.0)
        prob += (
            item.current_stock + Q[i] >= item.forecast_demand + safety_stock,
            f"ServiceLevel_{i}",
        )

    # ── Solve ──
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    logger.info(f"Optimization status: {status}")

    # ── Extract Results ──
    results = []
    for i, item in enumerate(items):
        reorder_qty = pulp.value(Q[i]) or 0.0
        safety_stock = safety_factor * item.forecast_std * np.sqrt(item.lead_time_days / 30.0)
        reorder_point = item.forecast_demand + safety_stock

        results.append({
            "category": item.category,
            "current_stock": round(item.current_stock, 0),
            "forecast_demand": round(item.forecast_demand, 0),
            "forecast_std": round(item.forecast_std, 1),
            "reorder_quantity": round(reorder_qty, 0),
            "reorder_point": round(reorder_point, 0),
            "safety_stock": round(safety_stock, 0),
            "total_inventory_after_order": round(item.current_stock + reorder_qty, 0),
            "unit_cost": round(item.unit_cost, 2),
            "holding_cost": round(holding_cost_pct * item.unit_cost * (item.current_stock / 2.0 + reorder_qty), 2),
        })

    result_df = pd.DataFrame(results)
    total_cost_value = pulp.value(prob.objective)

    return {
        "results": result_df,
        "total_cost": round(total_cost_value, 2) if total_cost_value else 0,
        "status": status,
    }


def run_optimization_pipeline(
    forecast_df: pd.DataFrame,
    cost_data: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Dict:
    """
    Run the full inventory optimization pipeline given forecast data.

    Args:
        forecast_df: DataFrame with columns [category, predicted_orders, ...]
        cost_data: Optional DataFrame with unit costs per category.
        **kwargs: Override optimization parameters.

    Returns:
        Optimization results dict.
    """
    logger.info("=" * 60)
    logger.info("Inventory Optimization Pipeline")
    logger.info("=" * 60)

    # Default costs if not provided
    if cost_data is None:
        cost_data = forecast_df.copy()
        cost_data["unit_cost"] = 50.0  # Default cost
        cost_data["selling_price"] = 80.0
        cost_data["storage_per_unit"] = 1.0
        cost_data["current_stock"] = 100.0

    items = []
    for _, row in forecast_df.iterrows():
        cat = row.get("category", row.get("product_category_name_english", "unknown"))
        cost_row = cost_data[cost_data["category"] == cat] if "category" in cost_data.columns else None

        unit_cost = float(cost_row["unit_cost"].values[0]) if cost_row is not None and len(cost_row) > 0 else 50.0
        selling_price = float(cost_row["selling_price"].values[0]) if cost_row is not None and len(cost_row) > 0 else 80.0
        storage = float(cost_row["storage_per_unit"].values[0]) if cost_row is not None and len(cost_row) > 0 else 1.0
        current_stock = float(cost_row["current_stock"].values[0]) if cost_row is not None and len(cost_row) > 0 else 100.0

        demand = row.get("predicted_orders", row.get(TARGET_COL, 0))
        demand_std = demand * 0.3  # Assume 30% CV if not available

        inv_input = InventoryInput(
            category=str(cat),
            forecast_demand=float(demand),
            forecast_std=float(demand_std),
            current_stock=float(current_stock),
            unit_cost=unit_cost,
            selling_price=selling_price,
            storage_per_unit=storage,
            lead_time_days=INVENTORY_PARAMS["lead_time_days"],
        )
        items.append(inv_input)

    results = solve_inventory_optimization(items, **kwargs)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    sample_items = [
        InventoryInput("bed_bath_table", 120, 20, 50, 45, 75, 7, 1.0),
        InventoryInput("health_beauty", 85, 15, 40, 55, 90, 7, 0.8),
        InventoryInput("sports_leisure", 60, 12, 30, 65, 100, 7, 1.2),
        InventoryInput("furniture_decor", 40, 10, 25, 80, 130, 14, 2.0),
        InventoryInput("computers_accessories", 95, 18, 60, 200, 350, 5, 0.5),
    ]

    result = solve_inventory_optimization(sample_items)
    print(f"Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:,.2f}")
    print("\nRecommendations:")
    print(result["results"].to_string(index=False))
