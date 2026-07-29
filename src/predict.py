"""
This script handles the inference part - making predictions using our trained models.
It generates forecasts for any category by loading the saved ensemble model
and building the required features for future dates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from src.config import (
    CATEGORY_COL, TARGET_COL, DATE_COL,
    MODELS_DIR, FORECAST_HORIZON,
)
from src.utils import load_model

logger = logging.getLogger(__name__)


def load_ensemble_for_category(category: str) -> Dict:
    """Load the saved ensemble model for a specific category."""
    try:
        results = load_model(f"ensemble_{category}.pkl")
        return results
    except FileNotFoundError:
        logger.error(f"No ensemble model found for category: {category}")
        return {}


def generate_future_dates(last_date: pd.Timestamp, days: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Create a dataframe with future dates starting from the day after last_date."""
    future_dates = pd.DataFrame({
        "date": pd.date_range(start=last_date + timedelta(days=1), periods=days, freq="D"),
    })
    return future_dates


def build_future_features(future_df: pd.DataFrame, category: str, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all the features needed for making predictions on future dates.
    Seasonal and holiday features are easy since we know the dates.
    For lag features, we use the last known values from history as approximations.
    """
    from src.features import add_seasonal_features, add_holiday_feature

    df = future_df.copy()
    df[CATEGORY_COL] = category
    df[TARGET_COL] = np.nan

    # These features are deterministic - we know the day of week, month, holidays etc.
    df = add_seasonal_features(df)
    df = add_holiday_feature(df)

    # For lag features, use the most recent historical values
    cat_history = history_df[history_df[CATEGORY_COL] == category].sort_values("date")
    if len(cat_history) > 0:
        last_known = cat_history.iloc[-1]
        df["order_count_lag_1"] = last_known["order_count"]
        df["order_count_lag_2"] = cat_history.iloc[-2]["order_count"] if len(cat_history) > 1 else 0
        df["order_count_lag_3"] = cat_history.iloc[-3]["order_count"] if len(cat_history) > 2 else 0
        df["order_count_lag_7"] = cat_history.iloc[-7]["order_count"] if len(cat_history) > 6 else 0

        # Rolling averages from the end of history
        if len(cat_history) >= 7:
            df["order_count_rolling_mean_7d"] = cat_history["order_count"].tail(7).mean()
            df["order_count_rolling_std_7d"] = cat_history["order_count"].tail(7).std()
        else:
            df["order_count_rolling_mean_7d"] = 0
            df["order_count_rolling_std_7d"] = 0

        if len(cat_history) >= 14:
            df["order_count_rolling_mean_14d"] = cat_history["order_count"].tail(14).mean()
            df["order_count_rolling_std_14d"] = cat_history["order_count"].tail(14).std()
        else:
            df["order_count_rolling_mean_14d"] = 0
            df["order_count_rolling_std_14d"] = 0

        if len(cat_history) >= 30:
            df["order_count_rolling_mean_30d"] = cat_history["order_count"].tail(30).mean()
            df["order_count_rolling_std_30d"] = cat_history["order_count"].tail(30).std()
        else:
            df["order_count_rolling_mean_30d"] = 0
            df["order_count_rolling_std_30d"] = 0

    df = df.fillna(0)
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get list of feature column names excluding identifiers."""
    from src.features import get_feature_columns
    return get_feature_columns(df)


def predict(
    category: str,
    days: int = FORECAST_HORIZON,
    results: Optional[Dict] = None,
    featured_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main prediction function.
    Takes a category name and number of days, then generates a forecast
    using the loaded ensemble model.

    Returns:
        - forecast_df: dataframe with daily predictions and confidence intervals
        - info: dictionary with summary statistics
    """
    if results is None:
        results = load_ensemble_for_category(category)

    if not results:
        raise ValueError(f"No model found for category: {category}")

    if featured_data is None:
        featured_data = load_model("featured_data.pkl")

    cat_history = featured_data[featured_data[CATEGORY_COL] == category].sort_values("date")

    if len(cat_history) == 0:
        raise ValueError(f"No historical data found for category: {category}")

    # Prepare future dates and features
    last_date = cat_history["date"].max()
    future_dates = generate_future_dates(last_date, days)
    future_features = build_future_features(future_dates, category, cat_history)

    cat_features = get_feature_cols(cat_history)

    for col in cat_features:
        if col not in future_features.columns:
            future_features[col] = 0

    # Get predictions from each model and combine them
    from src.train import predict_xgboost, predict_lstm, predict_prophet
    from src.config import ENSEMBLE_WEIGHTS

    weights = results.get("ensemble_weights", ENSEMBLE_WEIGHTS)
    predictions = {}

    if "xgboost" in results and results["xgboost"] is not None:
        xgb_model = results["xgboost"]["model"]
        predictions["xgboost"] = predict_xgboost(xgb_model, future_features, cat_features)

    if "prophet" in results and results["prophet"] is not None:
        prophet_model = results["prophet"]["model"]
        if prophet_model is not None:
            prophet_future = future_features.rename(columns={"date": "ds", TARGET_COL: "y"})
            predictions["prophet"] = predict_prophet(prophet_model, prophet_future)

    if "lstm" in results and results["lstm"] is not None:
        lstm_artifacts = results["lstm"]["model"]
        if lstm_artifacts is not None:
            predictions["lstm"] = predict_lstm(lstm_artifacts, cat_history, future_features, cat_features, category)

    # Weighted average ensemble
    ensemble_pred = np.zeros(len(future_features))
    weight_sum = 0.0

    for name, pred in predictions.items():
        w = weights.get(name, 0)
        ensemble_pred += w * pred
        weight_sum += w

    if weight_sum > 0:
        ensemble_pred /= weight_sum

    ensemble_pred = np.maximum(ensemble_pred, 0)

    # Build the output dataframe
    result_df = future_features[["date"]].copy()
    result_df["category"] = category
    result_df["predicted_orders"] = np.round(ensemble_pred).astype(int)
    result_df["predicted_lower"] = np.maximum(np.round(ensemble_pred * 0.85).astype(int), 0)
    result_df["predicted_upper"] = np.round(ensemble_pred * 1.15).astype(int)

    for name, preds in predictions.items():
        result_df[f"{name}_prediction"] = np.round(np.maximum(preds, 0)).astype(int)

    result_df = result_df.sort_values("date").reset_index(drop=True)

    # Calculate summary statistics
    recent_actuals = cat_history[
        cat_history["date"] >= cat_history["date"].max() - timedelta(days=30)
    ][TARGET_COL].values

    mean_demand = recent_actuals.mean() if len(recent_actuals) > 0 else 30

    info = {
        "category": category,
        "forecast_days": days,
        "total_forecast_orders": int(ensemble_pred.sum()),
        "avg_daily_forecast": round(float(ensemble_pred.mean()), 1),
        "avg_historical_daily": round(float(mean_demand), 1),
        "last_training_date": str(cat_history["date"].max().date()),
    }

    return result_df, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        cat = "bed_bath_table"
        results = load_ensemble_for_category(cat)
        if results:
            featured = load_model("featured_data.pkl")
            forecast, info = predict(cat, days=30, results=results, featured_data=featured)
            print(f"Forecast for {cat}: {info['total_forecast_orders']} orders over {info['forecast_days']} days")
        else:
            print("No model found. Run src/train.py first.")
    except Exception as e:
        print(f"Test failed: {e}")
