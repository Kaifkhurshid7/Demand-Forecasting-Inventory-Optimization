"""
Configuration module for demand forecasting and inventory optimization system.

This module centralizes all configuration parameters, paths, and constants used
across the ML pipeline, ensuring consistency and facilitating maintenance.

Author: Kaif Khurshid

"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# DATA DIRECTORIES
# ============================================================================
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Olist Brazilian E-Commerce dataset file mapping.
# These 9 files are required for the data ingestion pipeline.
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# ============================================================================
# MODEL & REPORTING DIRECTORIES
# ============================================================================
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================
# Column names for temporal and target variables
DATE_COL = "order_purchase_timestamp"
TARGET_COL = "order_count"
CATEGORY_COL = "product_category_name_english"

# Walk-forward validation split: ensures chronological consistency and
# prevents temporal data leakage (no future information in training).
TRAIN_START = "2016-09-01"
TRAIN_END = "2018-06-30"
VAL_START = "2018-07-01"
VAL_END = "2018-08-31"
TEST_START = "2018-09-01"
TEST_END = "2018-10-31"

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Lag features capture short-term temporal dependencies.
LAG_DAYS = [1, 2, 3, 7, 14, 21, 30]

# Rolling window statistics aggregate medium-term trends and volatility.
ROLLING_WINDOWS = [7, 14, 30]

# Brazilian public holidays for seasonal decomposition and modeling.
HOLIDAY_DATES = [
    "2016-11-15", "2016-12-25", "2017-01-01", "2017-02-25", "2017-04-14",
    "2017-05-01", "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02",
    "2017-11-15", "2017-12-25", "2018-01-01", "2018-02-10", "2018-03-30",
    "2018-05-01", "2018-05-31", "2018-09-07", "2018-10-12",
]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Prophet configuration for time series decomposition.
PROPHET_PARAMS = {
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "uncertainty_samples": 0,
}

# XGBoost gradient boosting regressor configuration.
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
    "verbosity": 0,
}

# LSTM recurrent neural network architecture and training parameters.
LSTM_PARAMS = {
    "seq_length": 30,
    "epochs": 50,
    "batch_size": 32,
    "lstm_units": [64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "patience": 10,
}

# Ensemble model weighting: determined inversely by validation RMSE.
# Higher-performing models receive greater weight in the final forecast.
ENSEMBLE_WEIGHTS = {"prophet": 0.25, "xgboost": 0.40, "lstm": 0.35}

# ============================================================================
# INVENTORY OPTIMIZATION PARAMETERS
# ============================================================================
INVENTORY_PARAMS = {
    "holding_cost_pct": 0.25,      # Annual inventory holding cost as % of unit cost
    "stockout_cost_pct": 0.40,     # Lost sales penalty as % of unit cost
    "service_level": 0.95,         # Target fill rate (95th percentile)
    "storage_capacity": 10000,     # Total warehouse capacity units
    "budget": 500000,              # Total procurement budget (currency)
    "lead_time_days": 7,           # Supplier lead time
}

# ============================================================================
# FORECASTING CONFIGURATION
# ============================================================================
FORECAST_HORIZON = 90  # Default prediction window in days

# ============================================================================
# RANDOM STATE
# ============================================================================
# Fixed seed for reproducibility across stochastic components.
RANDOM_STATE = 42
