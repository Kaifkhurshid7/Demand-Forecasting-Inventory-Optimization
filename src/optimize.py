"""
Inventory Optimization using Linear Programming (PuLP).
Solves for the optimal reorder quantities per category to minimize
holding costs and stockout costs, while respecting storage and budget limits.
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
    """
    Data class that holds all input information needed for optimization.
    Each item represents one product category with its demand forecast and costs.
    """
    category: str
    forecast_demand: float       # Expected demand for the period
    forecast_std: float          # Standard deviation of demand (uncertainty)
    current_stock: float         # How many units we currently have
    unit_cost: float             # Cost to purchase one unit
    selling_price: float         # Price we sell it for
    lead_time_days: int          # How long supplier takes to deliver
    storage_per_unit: float      # Space taken by one unit


def solve_inventory_optimization(
    items: List[InventoryInput],
    storage_capacity: Optional[float] = None,
    budget: Optional[float] = None,
    service_level: Optional[float] = None,
    holding_cost_pct: Optional[float] = None,
    stockout_cost_pct: Optional[float] = None,
) -> Dict:
    """
    Uses PuLP to find optimal reorder quantities that minimize total costs.

    The LP model works like this:
        - Decision: How many units to reorder for each category (Q_i)
        - Goal: Minimize (holding cost + stockout cost)
        - Rules: Stay within storage capacity, budget, and meet service level

    Args:
        items: List of product categories with their demand data
        storage_capacity: Total warehouse space available
        budget: Maximum money we can spend on new inventory
        service_level: Target probability of not running out of stock
        holding_cost_pct: Annual cost to hold inventory as % of unit cost
        stockout_cost_pct: Cost of lost sales as % of unit cost

    Returns:
        Dictionary with results dataframe, total cost, and solution status
    """
    import pulp
    from scipy.stats import norm

    # Use defaults from config if parameters not provided
    storage_capacity = storage_capacity or INVENTORY_PARAMS["storage_capacity"]
    budget = budget or INVENTORY_PARAMS["budget"]
    service_level = service_level or INVENTORY_PARAMS["service_level"]
    holding_cost_pct = holding_cost_pct or INVENTORY_PARAMS["holding_cost_pct"]
    stockout_cost_pct = stockout_cost_pct or INVENTORY_PARAMS["stockout_cost_pct"]

    # Z-score for the given service level (assuming normal distribution)
    safety_factor = norm.ppf(service_level)

    # Create the optimization problem (we want to minimize cost)
    prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

    # Decision variables: how much to reorder for each category
    n = len(items)
    Q = [pulp.LpVariable(f"Q_{i}", lowBound=0, cat="Continuous") for i in range(n)]

    # Objective: Minimize total cost = holding cost + stockout cost
    # Holding cost depends on average inventory level
    # Stockout cost happens when demand exceeds available stock
    # We linearize the stockout term using auxiliary variables S_i >= 0
    S = [pulp.LpVariable(f"S_{i}", lowBound=0, cat="Continuous") for i in range(n)]

    # S_i >= forecast_demand - (current_stock + Q_i), i.e., unmet demand
    for i, item in enumerate(items):
        prob += S[i] >= item.forecast_demand - (item.current_stock + Q[i])

    total_cost = pulp.lpSum([
        (
            holding_cost_pct * item.unit_cost * (item.current_stock / 2.0 + Q[i])
            + stockout_cost_pct * item.unit_cost * S[i]
        )
        for i, item in enumerate(items)
    ])
    prob += total_cost

    # Constraint 1: Total storage space used must be within capacity
    prob += pulp.lpSum([item.storage_per_unit * Q[i] for i, item in enumerate(items)]) <= storage_capacity

    # Constraint 2: Total purchase cost must be within budget
    prob += pulp.lpSum([item.unit_cost * Q[i] for i, item in enumerate(items)]) <= budget

    # Constraint 3: Maintain safety stock for service level
    # Current stock + reorder quantity must cover expected demand + safety stock
    for i, item in enumerate(items):
        safety_stock = safety_factor * item.forecast_std * np.sqrt(item.lead_time_days / 30.0)
        prob += (
            item.current_stock + Q[i] >= item.forecast_demand + safety_stock,
            f"ServiceLevel_{i}",
        )

    # Solve the LP problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    logger.info(f"Optimization status: {status}")

    # Extract the results
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


def run_optimization_pipeline(forecast_df: pd.DataFrame, **kwargs) -> Dict:
    """
    Run the full optimization pipeline using forecast data.
    Takes a dataframe with predicted orders and converts it into inventory inputs.

    Args:
        forecast_df: DataFrame with at least [category, predicted_orders] columns
        **kwargs: Additional parameters to pass to the solver

    Returns:
        Optimization results from solve_inventory_optimization()
    """
    logger.info("Running inventory optimization pipeline...")

    items = []
    for _, row in forecast_df.iterrows():
        cat = row.get("category", row.get("product_category_name_english", "unknown"))
        demand = row.get("predicted_orders", row.get(TARGET_COL, 0))

        inv_input = InventoryInput(
            category=str(cat),
            forecast_demand=float(demand),
            forecast_std=float(demand * 0.3),  # Assume 30% coefficient of variation
            current_stock=100.0,
            unit_cost=50.0,
            selling_price=80.0,
            storage_per_unit=1.0,
            lead_time_days=INVENTORY_PARAMS["lead_time_days"],
        )
        items.append(inv_input)

    results = solve_inventory_optimization(items, **kwargs)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data for 5 categories
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
