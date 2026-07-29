"""
Data Loader — Load, clean, and join all 9 Olist CSV files.

Outputs:
    - Unified order-level DataFrame (saved as Parquet)
    - Daily aggregated counts per category (saved as Parquet)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OLIST_FILES, DATE_COL, CATEGORY_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_csv(filename: str, dtype: Optional[Dict] = None, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load a single CSV from the raw data directory."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required file: {filepath}")
    logger.info(f"Loading {filepath}")
    df = pd.read_csv(filepath, dtype=dtype, parse_dates=parse_dates)
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_all_raw_data() -> Dict[str, pd.DataFrame]:
    """Load all 9 Olist CSV files into a dictionary of DataFrames."""
    raw = {}
    for key, filename in OLIST_FILES.items():
        if key in ("orders",):
            raw[key] = load_csv(filename, parse_dates=["order_purchase_timestamp",
                                                         "order_approved_at",
                                                         "order_delivered_carrier_date",
                                                         "order_delivered_customer_date",
                                                         "order_estimated_delivery_date"])
        elif key in ("reviews",):
            raw[key] = load_csv(filename, parse_dates=["review_creation_date",
                                                        "review_answer_timestamp"])
        else:
            raw[key] = load_csv(filename)
    return raw


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Clean orders DataFrame: filter unprocessed, drop duplicates, sort."""
    logger.info("Cleaning orders...")
    df = orders.copy()
    # Keep only delivered orders with a valid purchase timestamp
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=[DATE_COL])
    # Drop duplicates
    df = df.drop_duplicates(subset=["order_id"])
    # Sort chronologically
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    logger.info(f"  → {len(df):,} delivered orders retained")
    return df


def join_tables(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join all tables into a single unified order-level DataFrame.
    Each row represents one item in one order.
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

    # ── Merge order_items → orders ──
    df = orders.merge(items, on="order_id", how="inner")

    # ── Merge products ──
    df = df.merge(products, on="product_id", how="left")

    # ── Merge category translation ──
    df = df.merge(translation, on="product_category_name", how="left")

    # ── Merge customers ──
    df = df.merge(customers, on="customer_id", how="left")

    # ── Merge sellers ──
    df = df.merge(sellers, on="seller_id", how="left")

    # ── Merge payments (aggregate per order: total, installments, methods) ──
    pay_agg = payments.groupby("order_id").agg(
        payment_sequential_max=("payment_sequential", "max"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_sum=("payment_value", "sum"),
        payment_method_count=("payment_type", "nunique"),
        payment_type_credit_card=("payment_type", lambda x: (x == "credit_card").sum()),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    # ── Merge reviews (average score per order) ──
    rev_agg = reviews.groupby("order_id").agg(
        review_score_mean=("review_score", "mean"),
        review_score_count=("review_score", "count"),
    ).reset_index()
    df = df.merge(rev_agg, on="order_id", how="left")

    logger.info(f"  → Unified table: {len(df):,} rows, {len(df.columns)} columns")
    return df


def aggregate_daily_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily order counts per product category.

    Returns a DataFrame with columns:
        date, category, order_count, and additional aggregates.
    """
    logger.info("Aggregating daily counts per category...")

    # Ensure date column is datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["date"] = df[DATE_COL].dt.date

    # Fill missing categories
    df[CATEGORY_COL] = df[CATEGORY_COL].fillna("unknown").astype(str)

    # Daily order count per category
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

    logger.info(f"  → {len(daily):,} rows across {daily[CATEGORY_COL].nunique()} categories")
    return daily


def save_processed(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame to the processed directory as Parquet."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved → {path}")


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed Parquet file."""
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full data ingestion pipeline.
    Returns (unified_df, daily_category_df).
    """
    logger.info("=" * 60)
    logger.info("Starting Data Ingestion Pipeline")
    logger.info("=" * 60)

    raw = load_all_raw_data()
    unified = join_tables(raw)

    # Drop columns that are not useful for modeling
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

    logger.info("Pipeline complete.")
    return unified, daily


if __name__ == "__main__":
    run_pipeline()
