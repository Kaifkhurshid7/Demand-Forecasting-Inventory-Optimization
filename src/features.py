"""
Feature Engineering — Create lag, rolling window, seasonal, and exogenous features.

Transforms the daily category-level demand DataFrame into a supervised-learning
ready dataset with features for model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from src.config import (
    CATEGORY_COL, TARGET_COL, DATE_COL,
    LAG_DAYS, ROLLING_WINDOWS, HOLIDAY_DATES,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add lag features for the specified columns.

    For each category separately, creates t-n shifts.
    """
    logger.info("Adding lag features...")
    if columns is None:
        columns = [TARGET_COL]

    df = df.copy().sort_values(["date", CATEGORY_COL])
    result = df.copy()

    for col in columns:
        for lag in LAG_DAYS:
            feature_name = f"{col}_lag_{lag}"
            result[feature_name] = result.groupby(CATEGORY_COL)[col].shift(lag)

    logger.info(f"  → Added {len(LAG_DAYS) * len(columns)} lag features")
    return result


def add_rolling_features(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Add rolling window statistics (mean, std) per category."""
    logger.info("Adding rolling window features...")
    if columns is None:
        columns = [TARGET_COL]

    df = df.copy().sort_values(["date", CATEGORY_COL])
    result = df.copy()

    for col in columns:
        for window in ROLLING_WINDOWS:
            # Rolling mean
            result[f"{col}_rolling_mean_{window}d"] = (
                result.groupby(CATEGORY_COL)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            # Rolling std
            result[f"{col}_rolling_std_{window}d"] = (
                result.groupby(CATEGORY_COL)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
            )

    logger.info(f"  → Added rolling features for windows {ROLLING_WINDOWS}")
    return result


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal dummy features from the date column."""
    logger.info("Adding seasonal features...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["date"].dt.dayofweek
    # Month
    df["month"] = df["date"].dt.month
    # Quarter
    df["quarter"] = df["date"].dt.quarter
    # Day of month
    df["day_of_month"] = df["date"].dt.day
    # Week of year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Month start / end
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # ── One-hot encode day of week (drop first to avoid multicollinearity) ──
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow").astype(int)
    dow_dummies = dow_dummies.drop(columns=dow_dummies.columns[0])  # drop Monday
    df = pd.concat([df, dow_dummies], axis=1)

    # ── One-hot encode month ──
    month_dummies = pd.get_dummies(df["month"], prefix="month").astype(int)
    month_dummies = month_dummies.drop(columns=month_dummies.columns[0])  # drop January
    df = pd.concat([df, month_dummies], axis=1)

    logger.info(f"  → Added seasonal dummies")
    return df


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Brazilian public holidays."""
    logger.info("Adding holiday flags...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    holiday_set = set(pd.to_datetime(HOLIDAY_DATES).date)
    df["is_holiday"] = df["date"].dt.date.isin(holiday_set).astype(int)
    # Also add pre/post holiday indicators
    df["pre_holiday"] = df["date"].shift(-1).dt.date.isin(holiday_set).astype(int) if len(df) > 0 else 0
    df["post_holiday"] = df["date"].shift(1).dt.date.isin(holiday_set).astype(int) if len(df) > 0 else 0
    logger.info(f"  → {df['is_holiday'].sum():,} holiday dates flagged")
    return df


def add_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add exogenous / external features.
    These include aggregate price, freight, review score, payment features.
    """
    logger.info("Adding exogenous features...")
    df = df.copy()

    # Already have: avg_price, total_revenue, avg_freight, avg_review_score,
    # avg_installments — fill forward to handle missing days
    exog_cols = ["avg_price", "total_revenue", "avg_freight", "avg_review_score", "avg_installments"]
    existing = [c for c in exog_cols if c in df.columns]

    for col in existing:
        # Fill missing values per category using forward fill
        df[col] = df.groupby(CATEGORY_COL)[col].transform(lambda x: x.ffill().bfill())
        # Add lagged version
        for lag in [1, 7, 14]:
            df[f"{col}_lag_{lag}"] = df.groupby(CATEGORY_COL)[col].shift(lag)

    logger.info(f"  → Exogenous features added: {existing}")
    return df


def create_feature_pipeline(
    df: pd.DataFrame,
    fill_na: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on daily category data.
    """
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", CATEGORY_COL]).reset_index(drop=True)

    # Apply each feature group
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_seasonal_features(df)
    df = add_holiday_feature(df)
    df = add_exogenous_features(df)

    # Fill remaining NaN values (from lags / rolling at start of series)
    if fill_na:
        initial_len = len(df)
        # Drop rows where target is NaN
        df = df.dropna(subset=[TARGET_COL])
        # For feature columns, fill NaN with 0 (they are at series start)
        feature_cols = [c for c in df.columns if c not in [TARGET_COL, "date", CATEGORY_COL]]
        df[feature_cols] = df[feature_cols].fillna(0)
        logger.info(f"  → Filled NaN in {initial_len - len(df)} rows (target) and all feature NaN → 0")

    logger.info(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def train_val_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split.
    """
    logger.info("Splitting data chronologically...")
    df = df.copy().sort_values(["date", CATEGORY_COL])

    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()

    logger.info(f"  Train: {len(train):,} rows ({TRAIN_START} → {TRAIN_END})")
    logger.info(f"  Val:   {len(val):,} rows ({VAL_START} → {VAL_END})")
    logger.info(f"  Test:  {len(test):,} rows ({TEST_START} → {TEST_END})")

    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get the list of feature columns (exclude non-feature columns)."""
    exclude = {TARGET_COL, "date", CATEGORY_COL, "product_category_name"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from src.data_loader import load_processed
    daily = load_processed("daily_category_demand")
    featured = create_feature_pipeline(daily)
    train, val, test = train_val_test_split(featured)
    print(f"Features: {len(get_feature_columns(featured))}")
