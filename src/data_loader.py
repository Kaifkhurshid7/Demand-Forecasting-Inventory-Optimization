"""
This script handles loading and cleaning all 9 Olist CSV files.
It joins them into one big table and also creates daily aggregates by category.

Outputs:
    - unified_orders.parquet -> combined order-level data
    - daily_category_demand.parquet -> daily order counts per category
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OLIST_FILES, DATE_COL, CATEGORY_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_csv(filename: str, dtype: Optional[Dict] = None, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Read a single CSV file from the raw data folder."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    logger.info(f"Loading {filepath}")
    df = pd.read_csv(filepath, dtype=dtype, parse_dates=parse_dates)
    logger.info(f"  -> {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_all_raw_data() -> Dict[str, pd.DataFrame]:
    """Go through all 9 CSV files and load them into a dictionary."""
    raw = {}
    for key, filename in OLIST_FILES.items():
        if key == "orders":
            raw[key] = load_csv(filename, parse_dates=[
                "order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ])
        elif key == "reviews":
            raw[key] = load_csv(filename, parse_dates=[
                "review_creation_date", "review_answer_timestamp"
            ])
        else:
            raw[key] = load_csv(filename)
    return raw


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-delivered orders and clean up the dataframe."""
    logger.info("Cleaning orders...")
    df = orders.copy()
    # We only care about delivered orders with valid timestamps
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=[DATE_COL])
    df = df.drop_duplicates(subset=["order_id"])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    logger.info(f"  -> {len(df):,} delivered orders after cleaning")
    return df


def join_tables(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all the tables together into one big dataframe.
    Each row will represent one item in one order.
    """
    logger.info("Joining tables...")

    orders = clean_orders(raw["orders"])
    items = raw["order_items"].copy()
    products = raw["products"].copy()
    customers = raw["customers"].copy()
    payments = raw["payments"].copy()
    reviews = raw["reviews"].copy()
    sellers = raw["sellers"].copy()
    translation = raw["category_translation"].copy()

    # Start with orders and keep merging
    df = orders.merge(items, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(translation, on="product_category_name", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")

    # For payments, we aggregate per order since there can be multiple payment methods
    pay_agg = payments.groupby("order_id").agg(
        payment_sequential_max=("payment_sequential", "max"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_sum=("payment_value", "sum"),
        payment_method_count=("payment_type", "nunique"),
        payment_type_credit_card=("payment_type", lambda x: (x == "credit_card").sum()),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    # Average review score per order
    rev_agg = reviews.groupby("order_id").agg(
        review_score_mean=("review_score", "mean"),
        review_score_count=("review_score", "count"),
    ).reset_index()
    df = df.merge(rev_agg, on="order_id", how="left")

    logger.info(f"  -> Final table: {len(df):,} rows, {len(df.columns)} columns")
    return df


def aggregate_daily_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group orders by date and category to get daily demand counts.
    Also calculates average price, revenue, freight, and review scores.
    """
    logger.info("Creating daily category-level aggregates...")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["date"] = df[DATE_COL].dt.date

    # Handle any missing category names
    df[CATEGORY_COL] = df[CATEGORY_COL].fillna("unknown").astype(str)

    daily = (
        df.groupby(["date", CATEGORY_COL])
        .agg(
            order_count=("order_id", "nunique"),
            avg_price=("price", "mean"),
            total_revenue=("price", "sum"),
            avg_freight=("freight_value", "mean"),
            avg_review_score=("review_score_mean", "mean"),
            avg_installments=("payment_installments_max", "mean"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["date", CATEGORY_COL]).reset_index(drop=True)

    logger.info(f"  -> {len(daily):,} rows covering {daily[CATEGORY_COL].nunique()} categories")
    return daily


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Save processed dataframe as parquet file."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved: {path}")


def load_processed(name: str) -> pd.DataFrame:
    """Read a processed parquet file back."""
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete data ingestion pipeline.
    Returns the unified table and the daily category aggregates.
    """
    logger.info("Starting data ingestion pipeline...")

    raw = load_all_raw_data()
    unified = join_tables(raw)

    # Drop columns that are not useful for forecasting
    drop_cols = [
        "order_status", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
        "customer_id", "customer_unique_id", "customer_zip_code_prefix",
        "seller_zip_code_prefix", "product_description_length",
        "product_photos_qty", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ]
    drop_cols = [c for c in drop_cols if c in unified.columns]
    unified = unified.drop(columns=drop_cols)

    daily = aggregate_daily_category(unified)

    save_processed(unified, "unified_orders")
    save_processed(daily, "daily_category_demand")

    logger.info("Data pipeline finished!")
    return unified, daily


if __name__ == "__main__":
    run_pipeline()
