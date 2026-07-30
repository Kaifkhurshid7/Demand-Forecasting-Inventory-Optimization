"""
FastAPI inference service for demand forecasting and inventory optimization.

This module provides REST API endpoints for:
- Real-time demand forecasting by product category
- Inventory optimization recommendations via linear programming
- Health checks and service metadata

The service loads pre-trained ensemble models (XGBoost, Prophet, LSTM) and
featured datasets from the models/ directory for low-latency inference.

Endpoints:
    GET  /health            → Service health and available categories
    GET  /predict           → Demand forecast for specified category
    POST /optimize          → Inventory optimization recommendations

Author: Kaif Khurshid

"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODELS_DIR, CATEGORY_COL
from src.utils import load_model
from src.predict import predict as generate_forecast

from api.schemas import (
    PredictRequest, PredictResponse, PredictError,
    DailyForecast, OptimizeRequest, OptimizeResponse,
    InventoryRecommendation, HealthResponse, OptimizeError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description="Production inference service for demand forecasting and inventory planning",
    version="1.0.0",
)

# Enable CORS for cross-origin requests from dashboard and third-party clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL CACHING LAYER
# ============================================================================
class ModelCache:
    """
    Lazy-loaded cache for model artifacts and featured datasets.
    
    Caching strategy:
    - Avoids repeated disk I/O and deserialization on each request
    - Models loaded on first access and retained in memory
    - Significantly reduces inference latency after warm-up
    """
    def __init__(self):
        self._featured_data = None
        self._ensemble_models: Dict[str, Dict] = {}
        self._categories: List[str] = []

    @property
    def featured_data(self):
        """Load featured dataset containing historical features and targets."""
        if self._featured_data is None:
            self._featured_data = load_model("featured_data.pkl")
            logger.info("Featured data loaded into cache")
        return self._featured_data

    def get_ensemble(self, category: str) -> Dict:
        """Load ensemble model for specified product category."""
        if category not in self._ensemble_models:
            try:
                model_path = MODELS_DIR / f"ensemble_{category}.pkl"
                if not model_path.exists():
                    raise FileNotFoundError(f"No model for category: {category}")
                self._ensemble_models[category] = load_model(f"ensemble_{category}.pkl")
                logger.info(f"Loaded ensemble model: {category}")
            except FileNotFoundError:
                raise
        return self._ensemble_models[category]

    @property
    def categories(self):
        """Discover available product categories from saved model files."""
        if not self._categories:
            model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
            self._categories = sorted([
                f.stem.replace("ensemble_", "")
                for f in model_files
            ])
        return self._categories


cache = ModelCache()


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================
@app.on_event("startup")
async def startup():
    """
    Pre-load models and featured data on service startup.
    
    Warm-caching during startup reduces first-request latency.
    """
    try:
        _ = cache.featured_data
        logger.info(f"Startup complete. {len(cache.categories)} categories available.")
    except Exception as e:
        logger.warning(f"Startup cache warning: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Service metadata and available endpoints."""
    return {
        "name": "Demand Forecasting & Inventory Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check and available categories",
            "GET /predict": "Forecast demand (params: category, days)",
            "POST /optimize": "Inventory optimization recommendations",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Service health check endpoint.
    
    Returns operational status and list of available product categories.
    Useful for monitoring and load balancer health probes.
    """
    cats = cache.categories
    return HealthResponse(
        status="healthy",
        models_loaded=len(cats),
        categories_available=cats[:20],
    )


@app.get(
    "/predict",
    response_model=PredictResponse,
    responses={404: {"model": PredictError}, 500: {"model": PredictError}},
    tags=["Forecasting"],
)
async def predict(
    category: str = Query(..., description="Product category name"),
    days: int = Query(30, ge=1, le=365, description="Forecast horizon in days"),
):
    """
    Generate demand forecast for a product category.
    
    Generates daily predicted order counts with confidence intervals (lower/upper bounds)
    for the specified forecasting horizon. Forecasts are produced using a weighted
    ensemble of XGBoost, Prophet, and LSTM models, with weights determined by
    validation-set performance (RMSE).
    
    Args:
        category: Product category identifier (e.g., 'bed_bath_table')
        days: Number of days to forecast into the future (1-365)
        
    Returns:
        PredictResponse containing:
        - forecast: List of daily predictions with confidence intervals
        - Aggregate statistics: total orders, averages, training date
        
    Raises:
        HTTPException 404: Category not found in trained models
        HTTPException 500: Inference error (data mismatch, model corruption)
    """
    try:
        # Load pre-trained ensemble for category
        ensemble = cache.get_ensemble(category)
        featured = cache.featured_data

        # Generate forecast using weighted ensemble
        forecast_df, info = generate_forecast(
            category=category,
            days=days,
            results=ensemble,
            featured_data=featured,
        )

        # Format output response
        forecast_list = []
        for _, row in forecast_df.iterrows():
            forecast_list.append(DailyForecast(
                date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                predicted_orders=int(row["predicted_orders"]),
                predicted_lower=int(row["predicted_lower"]),
                predicted_upper=int(row["predicted_upper"]),
            ))

        return PredictResponse(
            status="success",
            category=category,
            forecast_days=days,
            total_forecast_orders=info["total_forecast_orders"],
            avg_daily_forecast=info["avg_daily_forecast"],
            avg_historical_daily=info["avg_historical_daily"],
            last_training_date=info["last_training_date"],
            forecast=forecast_list,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for category '{category}'. Available: {cache.categories[:10]}. Error: {e}",
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    responses={500: {"model": OptimizeError}},
    tags=["Optimization"],
)
async def optimize(req: OptimizeRequest):
    """
    Compute inventory optimization recommendations.
    
    Solves a linear program (via PuLP CBC solver) to determine optimal reorder
    quantities for each product category, minimizing total costs (holding + stockout)
    subject to:
    - Storage capacity constraint
    - Budget constraint
    - Service level target (95th percentile, normal distribution)
    
    The optimization uses demand forecasts from the ensemble model and cost
    parameters from config or request payload.
    """
    try:
        from src.optimize import run_optimization_pipeline

        # Use default categories if not provided
        if not req.categories:
            featured = cache.featured_data
            if CATEGORY_COL in featured.columns:
                recent = featured.sort_values("date").groupby(CATEGORY_COL).last().reset_index()
                forecast_df = recent.rename(columns={
                    CATEGORY_COL: "category",
                    "order_count": "predicted_orders",
                })
            else:
                raise ValueError("No featured data available and no categories provided.")
        else:
            forecast_df = None

        # Build optimization parameters from request or defaults
        opt_kwargs = {}
        if req.storage_capacity is not None:
            opt_kwargs["storage_capacity"] = req.storage_capacity
        if req.budget is not None:
            opt_kwargs["budget"] = req.budget
        if req.service_level is not None:
            opt_kwargs["service_level"] = req.service_level

        # Run LP solver
        result = run_optimization_pipeline(forecast_df, **opt_kwargs)

        # Format recommendations
        recommendations = [
            InventoryRecommendation(**row)
            for _, row in result["results"].iterrows()
        ]

        return OptimizeResponse(
            status="success",
            total_cost=result["total_cost"],
            recommendations=recommendations,
        )

    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
