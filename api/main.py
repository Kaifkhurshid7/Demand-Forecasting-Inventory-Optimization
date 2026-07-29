"""
FastAPI application that serves our demand forecasting models.
Provides endpoints for getting forecasts and inventory optimization recommendations.

Endpoints:
    GET  /          -> API information
    GET  /health    -> Check if API is running and what models are available
    GET  /predict   -> Get demand forecast for a category
    POST /optimize  -> Get inventory optimization suggestions
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description="Get demand forecasts and optimize inventory for e-commerce categories",
    version="1.0.0",
)

# Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cache for loaded models (lazy loading to save memory)
class ModelCache:
    """Loads and caches models so we don't read from disk every time."""
    def __init__(self):
        self._featured_data = None
        self._ensemble_models: Dict[str, Dict] = {}
        self._categories: List[str] = []

    @property
    def featured_data(self):
        if self._featured_data is None:
            self._featured_data = load_model("featured_data.pkl")
            logger.info("Featured data loaded into cache")
        return self._featured_data

    def get_ensemble(self, category: str) -> Dict:
        if category not in self._ensemble_models:
            model_path = MODELS_DIR / f"ensemble_{category}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"No model for category: {category}")
            self._ensemble_models[category] = load_model(f"ensemble_{category}.pkl")
            logger.info(f"Loaded ensemble model: {category}")
        return self._ensemble_models[category]

    @property
    def categories(self):
        if not self._categories:
            model_files = list(MODELS_DIR.glob("ensemble_*.pkl"))
            self._categories = sorted([
                f.stem.replace("ensemble_", "") for f in model_files
            ])
        return self._categories


cache = ModelCache()


@app.on_event("startup")
async def startup():
    """Pre-load data when the API starts up."""
    try:
        _ = cache.featured_data
        logger.info(f"Startup complete. {len(cache.categories)} categories loaded.")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")


@app.get("/", tags=["Info"])
async def root():
    """Show available endpoints and basic info."""
    return {
        "name": "Demand Forecasting & Inventory Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Check API health and available categories",
            "GET /predict": "Get demand forecast (params: category, days)",
            "POST /optimize": "Get inventory optimization suggestions",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check - shows if API is running and what models are available."""
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
    category: str = Query(..., description="Product category name (e.g. bed_bath_table)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
):
    """
    Generate a demand forecast for any product category.

    Returns daily predicted order counts with lower and upper confidence bounds
    for the specified number of days.
    """
    try:
        ensemble = cache.get_ensemble(category)
        featured = cache.featured_data

        forecast_df, info = generate_forecast(
            category=category,
            days=days,
            results=ensemble,
            featured_data=featured,
        )

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
            detail=f"No model for '{category}'. Available: {cache.categories[:10]}. Error: {e}",
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
    Get inventory optimization recommendations.

    Provide cost data per category and the solver will calculate
    optimal reorder quantities to minimize holding and stockout costs.
    """
    try:
        from src.optimize import run_optimization_pipeline

        if not req.categories:
            featured = cache.featured_data
            if CATEGORY_COL in featured.columns:
                recent = featured.sort_values("date").groupby(CATEGORY_COL).last().reset_index()
                forecast_df = recent.rename(columns={
                    CATEGORY_COL: "category",
                    "order_count": "predicted_orders",
                })
            else:
                raise ValueError("No featured data and no categories provided.")
        else:
            forecast_df = None

        opt_kwargs = {}
        if req.storage_capacity is not None:
            opt_kwargs["storage_capacity"] = req.storage_capacity
        if req.budget is not None:
            opt_kwargs["budget"] = req.budget
        if req.service_level is not None:
            opt_kwargs["service_level"] = req.service_level

        result = run_optimization_pipeline(forecast_df, **opt_kwargs)

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
