"""
Configuration file for the Demand Forecasting project.
Contains all the paths, parameters, and constants used across the project.
"""

import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# All 9 Olist CSV files that make up the dataset
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

# Model and reports directories
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Column names used throughout the project
DATE_COL = "order_purchase_timestamp"
TARGET_COL = "order_count"
CATEGORY_COL = "product_category_name_english"

# Date ranges for train/validation/test split (chronological order)
TRAIN_START = "2016-09-01"
TRAIN_END = "2018-06-30"
VAL_START = "2018-07-01"
VAL_END = "2018-08-31"
TEST_START = "2018-09-01"
TEST_END = "2018-10-31"

# Feature engineering parameters
LAG_DAYS = [1, 2, 3, 7, 14, 21, 30]  # How many days back to look for lag features
ROLLING_WINDOWS = [7, 14, 30]  # Window sizes for rolling averages

# Brazilian public holidays that might affect order patterns
HOLIDAY_DATES = [
    "2016-11-15",  # Republic Day
    "2016-12-25",  # Christmas
    "2017-01-01",  # New Year
    "2017-02-25",  # Carnaval
    "2017-04-14",  # Good Friday
    "2017-05-01",  # Labor Day
    "2017-06-15",  # Corpus Christi
    "2017-09-07",  # Independence Day
    "2017-10-12",  # Our Lady Aparecida
    "2017-11-02",  # All Souls' Day
    "2017-11-15",  # Republic Day
    "2017-12-25",  # Christmas
    "2018-01-01",  # New Year
    "2018-02-10",  # Carnaval
    "2018-03-30",  # Good Friday
    "2018-05-01",  # Labor Day
    "2018-05-31",  # Corpus Christi
    "2018-09-07",  # Independence Day
    "2018-10-12",  # Our Lady Aparecida
]

# Prophet model hyperparameters
PROPHET_PARAMS = {
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "uncertainty_samples": 0,
}

# XGBoost model hyperparameters
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
    "verbosity": 0,
}

# LSTM model hyperparameters
LSTM_PARAMS = {
    "seq_length": 30,
    "epochs": 50,
    "batch_size": 32,
    "lstm_units": [64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "patience": 10,
}

# Default weights for the ensemble model
ENSEMBLE_WEIGHTS = {"prophet": 0.25, "xgboost": 0.40, "lstm": 0.35}

# Inventory optimization default parameters
INVENTORY_PARAMS = {
    "holding_cost_pct": 0.25,       # 25% annual holding cost
    "stockout_cost_pct": 0.40,      # 40% stockout cost
    "service_level": 0.95,          # 95% service level target
    "storage_capacity": 10000,
    "budget": 500000,
    "lead_time_days": 7,
}

# Default forecast horizon (in days)
FORECAST_HORIZON = 90
RANDOM_STATE = 42
