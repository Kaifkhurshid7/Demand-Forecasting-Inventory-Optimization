"""
Training pipeline — Train Prophet, XGBoost, LSTM, and Ensemble models.

Uses walk-forward validation with a fixed train/val/test split
(chronological) to evaluate out-of-sample performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

from src.config import (
    CATEGORY_COL, TARGET_COL, DATE_COL,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    PROPHET_PARAMS, XGB_PARAMS, LSTM_PARAMS, ENSEMBLE_WEIGHTS,
    MODELS_DIR, RANDOM_STATE,
)
from src.features import (
    create_feature_pipeline, train_val_test_split,
    get_feature_columns,
)
from src.utils import (
    compute_metrics, save_model, save_metrics,
    plot_forecast_vs_actual, plot_residuals,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prophet Model
# ──────────────────────────────────────────────

def train_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    category: str,
):
    """Train Prophet on a single category (skipped on Windows due to CmdStanPy issues)."""
    logger.info(f"Training Prophet for category: {category} (SKIPPED - CmdStanPy compatibility issue)")
    
    # Return dummy results - will be replaced by XGBoost/LSTM
    train = train_df[train_df[CATEGORY_COL] == category].copy()
    val = val_df[val_df[CATEGORY_COL] == category].copy()
    
    y_true = val["order_count"].values
    y_pred = np.mean(train["order_count"].values) * np.ones_like(y_true)
    
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"  → Validation MAPE (using baseline): {metrics['mape']:.2f}%")
    
    return None, metrics, y_true, y_pred


def predict_prophet(model, future_df: pd.DataFrame) -> np.ndarray:
    """Generate predictions using a trained Prophet model."""
    if model is None:
        # Return baseline prediction
        return np.ones(len(future_df)) * 100
    pred_df = model.predict(future_df)
    return pred_df["yhat"].values


# ──────────────────────────────────────────────
# XGBoost Model
# ──────────────────────────────────────────────

def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_features: List[str],
    category: str,
):
    """Train XGBoost regressor."""
    import xgboost as xgb

    logger.info(f"Training XGBoost for category: {category}")

    train = train_df[train_df[CATEGORY_COL] == category].copy()
    val = val_df[val_df[CATEGORY_COL] == category].copy()

    X_train = train[cat_features].values
    y_train = train[TARGET_COL].values
    X_val = val[cat_features].values
    y_val = val[TARGET_COL].values

    model = xgb.XGBRegressor(
        **XGB_PARAMS,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    logger.info(f"  → Validation MAPE: {metrics['mape']:.2f}%")

    return model, metrics, y_val, y_pred


def predict_xgboost(model, features_df: pd.DataFrame, cat_features: List[str]) -> np.ndarray:
    """Generate predictions using a trained XGBoost model."""
    X = features_df[cat_features].values
    return model.predict(X)


# ──────────────────────────────────────────────
# LSTM Model
# ──────────────────────────────────────────────

def _build_lstm_model(seq_length: int, n_features: int) -> "tf.keras.Model":
    """Build and compile an LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(LSTM(LSTM_PARAMS["lstm_units"][0], return_sequences=True,
                   input_shape=(seq_length, n_features)))
    model.add(Dropout(LSTM_PARAMS["dropout"]))
    model.add(LSTM(LSTM_PARAMS["lstm_units"][1], return_sequences=False))
    model.add(Dropout(LSTM_PARAMS["dropout"]))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=LSTM_PARAMS["learning_rate"]),
        loss="mse",
    )
    return model


def _create_sequences(data: np.ndarray, seq_length: int):
    """Create sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_features: List[str],
    category: str,
):
    """Train an LSTM model."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    logger.info(f"Training LSTM for category: {category}")

    train = train_df[train_df[CATEGORY_COL] == category].copy()
    val = val_df[val_df[CATEGORY_COL] == category].copy()

    # Combine train + val for sufficient sequence data, but evaluate on val portion
    full_df = pd.concat([train, val], axis=0).reset_index(drop=True)
    full_data = full_df[cat_features + [TARGET_COL]].values

    # Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(full_data)

    # Create sequences
    seq_length = LSTM_PARAMS["seq_length"]
    X, y = _create_sequences(scaled_data, seq_length)

    # Split back into train/val
    train_len = len(train) - seq_length
    X_train, X_val_seq = X[:train_len], X[train_len:]
    y_train, y_val_seq = y[:train_len], y[train_len:]

    if len(X_train) < 10 or len(X_val_seq) < 5:
        logger.warning(f"  → Insufficient data for LSTM. Skipping.")
        return None, None, None, None

    n_features = scaled_data.shape[1]
    model = _build_lstm_model(seq_length, n_features)

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor="loss",
        patience=LSTM_PARAMS["patience"],
        restore_best_weights=True,
    )

    model.fit(
        X_train, y_train,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )

    # Predict on val sequences (predict target value only, last column)
    y_pred_scaled = model.predict(X_val_seq, verbose=0).flatten()

    # Inverse transform: we need to reconstruct the full matrix
    y_pred_full = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
    y_pred_full[:, -1] = y_pred_scaled
    y_pred_full = scaler.inverse_transform(y_pred_full)
    y_pred = y_pred_full[:, -1]

    # Inverse transform actuals similarly
    y_true_full = np.zeros((len(y_val_seq), scaled_data.shape[1]))
    y_true_full[:, -1] = y_val_seq
    y_true_full = scaler.inverse_transform(y_true_full)
    y_true = y_true_full[:, -1]

    y_pred = np.maximum(y_pred, 0)  # clip negative predictions
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"  → Validation MAPE: {metrics['mape']:.2f}%")

    return {"model": model, "scaler": scaler, "seq_length": seq_length, "n_features": n_features}, metrics, y_true, y_pred


def predict_lstm(lstm_artifacts, full_history_df: pd.DataFrame, future_df: pd.DataFrame, cat_features: List[str], category: str) -> np.ndarray:
    """Generate predictions using trained LSTM."""
    model = lstm_artifacts["model"]
    scaler = lstm_artifacts["scaler"]
    seq_length = lstm_artifacts["seq_length"]

    # Combine history + future
    cat_data = full_history_df[full_history_df[CATEGORY_COL] == category]
    seq_data = cat_data[cat_features + [TARGET_COL]].values

    # Pad with last known values for future
    last_known = seq_data[-1:]
    for _ in range(len(future_df)):
        padded = np.zeros((1, seq_data.shape[1]))
        padded[0, :-1] = seq_data[-1, :-1]  # use last features
        seq_data = np.vstack([seq_data, padded])

    scaled = scaler.transform(seq_data)
    # Take last len(future_df) sequences
    X = []
    for i in range(len(future_df)):
        start = len(scaled) - len(future_df) - seq_length + i
        X.append(scaled[start : start + seq_length])
    X = np.array(X)

    y_pred_scaled = model.predict(X, verbose=0).flatten()
    y_pred_full = np.zeros((len(y_pred_scaled), scaled.shape[1]))
    y_pred_full[:, -1] = y_pred_scaled
    y_pred_full = scaler.inverse_transform(y_pred_full)
    return np.maximum(y_pred_full[:, -1], 0)


# ──────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────

def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_features: List[str],
    category: str,
) -> Dict:
    """
    Train all models for a category and return the ensemble.
    Returns a dict with each model's artifacts, metrics, and the ensemble weights.
    """
    logger.info(f"{'='*50}")
    logger.info(f"Training Ensemble for category: {category}")
    logger.info(f"{'='*50}")

    results = {}

    # Prophet (skip with None on Windows)
    logger.info("\n--- Prophet ---")
    try:
        prophet_model, prophet_metrics, _, _ = train_prophet(train_df, val_df, category)
        results["prophet"] = {"model": prophet_model, "metrics": prophet_metrics}
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        results["prophet"] = None

    # XGBoost
    logger.info("\n--- XGBoost ---")
    try:
        xgb_model, xgb_metrics, _, _ = train_xgboost(train_df, val_df, cat_features, category)
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}
    except Exception as e:
        logger.warning(f"XGBoost training failed: {e}")
        results["xgboost"] = None

    # LSTM
    logger.info("\n--- LSTM ---")
    try:
        lstm_artifacts, lstm_metrics, _, _ = train_lstm(train_df, val_df, cat_features, category)
        if lstm_artifacts is not None:
            results["lstm"] = {"model": lstm_artifacts, "metrics": lstm_metrics}
    except Exception as e:
        logger.warning(f"LSTM training failed: {e}")
        results["lstm"] = None

    # ── Compute ensemble weights based on validation performance ──
    # Lower RMSE → higher weight
    rmse_values = {}
    for name, res in results.items():
        if res is not None and "metrics" in res:
            rmse_values[name] = res["metrics"]["rmse"]

    if rmse_values:
        inv_rmse = {k: 1.0 / max(v, 0.001) for k, v in rmse_values.items()}
        total_inv = sum(inv_rmse.values())
        weights = {k: v / total_inv for k, v in inv_rmse.items()}
    else:
        weights = {k: v for k, v in ENSEMBLE_WEIGHTS.items() if results.get(k) is not None}
        if weights:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {"xgboost": 1.0}

    results["ensemble_weights"] = weights

    logger.info(f"\nEnsemble weights: {weights}")
    return results


def predict_ensemble(
    ensemble_results: Dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_features: List[str],
    category: str,
) -> np.ndarray:
    """Generate ensemble predictions on test data."""
    weights = ensemble_results.get("ensemble_weights", ENSEMBLE_WEIGHTS)
    predictions = {}

    df_for_pred = val_df[val_df[CATEGORY_COL] == category]
    df_future = test_df[test_df[CATEGORY_COL] == category]

    if len(df_future) == 0:
        return np.array([])

    if "prophet" in ensemble_results and ensemble_results["prophet"] is not None:
        prophet_model = ensemble_results["prophet"]["model"]
        if prophet_model is not None:
            future = df_future.rename(columns={"date": "ds", TARGET_COL: "y"})
            predictions["prophet"] = predict_prophet(prophet_model, future)

    if "xgboost" in ensemble_results and ensemble_results["xgboost"] is not None:
        xgb_model = ensemble_results["xgboost"]["model"]
        predictions["xgboost"] = predict_xgboost(xgb_model, df_future, cat_features)

    if "lstm" in ensemble_results and ensemble_results["lstm"] is not None and "lstm" in weights:
        lstm_artifacts = ensemble_results["lstm"]["model"]
        predictions["lstm"] = predict_lstm(lstm_artifacts, df_for_pred, df_future, cat_features, category)

    # Weighted ensemble
    if not predictions:
        return np.zeros(len(df_future))

    ensemble_pred = np.zeros(len(df_future))
    weight_sum = 0.0

    for name, pred in predictions.items():
        w = weights.get(name, 0)
        ensemble_pred += w * pred
        weight_sum += w

    if weight_sum > 0:
        ensemble_pred /= weight_sum

    return np.maximum(ensemble_pred, 0)


# ──────────────────────────────────────────────
# Full Training Pipeline
# ──────────────────────────────────────────────

def train_all_categories(
    daily_df: pd.DataFrame,
    top_n_categories: Optional[int] = None,
) -> Dict:
    """
    Train ensemble models for each product category.

    Args:
        daily_df: Daily category-level demand DataFrame.
        top_n_categories: If set, only train on top N categories by volume.

    Returns:
        Dict mapping category name → ensemble results.
    """
    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Feature engineering
    featured = create_feature_pipeline(daily_df)
    train_df, val_df, test_df = train_val_test_split(featured)
    cat_features = get_feature_columns(featured)

    # Select categories
    if top_n_categories:
        top_cats = (
            train_df.groupby(CATEGORY_COL)[TARGET_COL]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_categories)
            .index
            .tolist()
        )
    else:
        top_cats = train_df[CATEGORY_COL].unique().tolist()

    logger.info(f"Training on {len(top_cats)} categories")

    all_results = {}
    for category in top_cats:
        try:
            results = train_ensemble(train_df, val_df, cat_features, category)
            all_results[category] = results

            # Predict on test set
            logger.info(f"\nPredicting on test set for: {category}")
            y_pred = predict_ensemble(results, val_df, test_df, cat_features, category)
            y_true = test_df[test_df[CATEGORY_COL] == category][TARGET_COL].values

            if len(y_true) == len(y_pred) and len(y_true) > 0:
                test_metrics = compute_metrics(y_true, y_pred)
                for name in ["prophet", "xgboost", "lstm"]:
                    if name in results:
                        all_results[category][name]["test_metrics"] = test_metrics
                all_results[category]["ensemble_test_metrics"] = test_metrics
                logger.info(f"  → Ensemble Test MAPE: {test_metrics['mape']:.2f}%")
        except Exception as e:
            logger.error(f"Failed for category {category}: {e}")
            continue

    # Save aggregate metrics
    aggregate_metrics = {}
    for cat, res in all_results.items():
        if "ensemble_test_metrics" in res:
            aggregate_metrics[cat] = res["ensemble_test_metrics"]

    if aggregate_metrics:
        avg_mape = np.mean([m["mape"] for m in aggregate_metrics.values()])
        avg_rmse = np.mean([m["rmse"] for m in aggregate_metrics.values()])
        logger.info(f"\n{'='*60}")
        logger.info(f"Aggregate Test Results ({len(aggregate_metrics)} categories):")
        logger.info(f"  Average MAPE: {avg_mape:.2f}%")
        logger.info(f"  Average RMSE: {avg_rmse:.2f}")
        save_metrics(aggregate_metrics, "all_categories")

    # Save full dataset for prediction use
    save_model(featured, "featured_data.pkl")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    from src.data_loader import load_processed
    daily = load_processed("daily_category_demand")

    # Train on top 5 categories for initial benchmark
    results = train_all_categories(daily, top_n_categories=5)
    print(f"Trained {len(results)} categories successfully.")
