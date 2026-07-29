"""
Utility functions used across the project.
Includes evaluation metrics (MAPE, RMSE, MAE), plotting helpers,
and functions for saving/loading models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path

from src.config import FIGURES_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


# Evaluation Metrics Functions

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE. Filters out zero values in actuals to avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE (Mean Absolute Error)."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Get all three evaluation metrics in one function call."""
    return {
        "mape": round(mean_absolute_percentage_error(y_true, y_pred), 2),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
    }


# Plotting Functions

def set_plot_style():
    """Apply consistent styling to all matplotlib plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })


def plot_time_series(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "order_count",
    title: str = "Daily Order Count",
    save_path: Optional[str] = None,
):
    """Plot a time series with a 7-day moving average overlay."""
    set_plot_style()
    fig, ax = plt.subplots()

    ax.plot(df[date_col], df[value_col], alpha=0.6, linewidth=0.8, label="Daily")
    if len(df) >= 7:
        ma = df[value_col].rolling(7, min_periods=1).mean()
        ax.plot(df[date_col], ma, linewidth=1.5, color="red", label="7-day MA")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Order Count")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(FIGURES_DIR / save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
    return fig


def plot_forecast_vs_actual(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Compare predicted values against actual values with error shading."""
    set_plot_style()
    fig, ax = plt.subplots()

    ax.plot(dates, y_true, label="Actual", linewidth=1.5, color="black")
    ax.plot(dates, y_pred, label=f"{model_name} Forecast", linewidth=1.2, alpha=0.8)
    ax.fill_between(dates, y_pred, y_true, alpha=0.15, color="red", label="Error")

    metrics = compute_metrics(y_true, y_pred)
    ax.set_title(title or f"{model_name} - Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Order Count")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(FIGURES_DIR / save_path, dpi=150, bbox_inches="tight")
    return fig, metrics


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
):
    """Plot residuals over time and their distribution histogram."""
    set_plot_style()
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"{model_name} - Residuals")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=30, edgecolor="white", alpha=0.7)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"{model_name} - Residual Distribution")
    axes[1].set_xlabel("Residual")

    fig.tight_layout()

    if save_path:
        fig.savefig(FIGURES_DIR / save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_category_comparison(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart showing top categories by total orders."""
    set_plot_style()
    cat_totals = df.groupby("product_category_name_english")["order_count"].sum().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(cat_totals)), cat_totals.values[::-1], color="steelblue")
    ax.set_yticks(range(len(cat_totals)))
    ax.set_yticklabels(cat_totals.index[::-1])
    ax.set_title(f"Top {top_n} Categories by Total Orders")
    ax.set_xlabel("Total Orders")
    fig.tight_layout()

    if save_path:
        fig.savefig(FIGURES_DIR / save_path, dpi=150, bbox_inches="tight")
    return fig


# Model Persistence Functions

def save_model(artifact, name: str):
    """Save a trained model to disk. Handles both sklearn/joblib and Keras formats."""
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name

    if name.endswith(".h5") or name.endswith(".keras"):
        artifact.save(str(path))
    else:
        joblib.dump(artifact, path)

    logger.info(f"Model saved: {path}")


def load_model(name: str):
    """Load a previously saved model from disk."""
    import joblib
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    if name.endswith(".h5") or name.endswith(".keras"):
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(str(path))
    else:
        return joblib.load(path)


def save_metrics(metrics: Dict, name: str):
    """Save evaluation metrics as a JSON file."""
    path = MODELS_DIR / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {path}")
