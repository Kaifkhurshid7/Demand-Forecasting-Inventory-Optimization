"""
Pydantic models for the FastAPI endpoints.
These define what data the API expects and returns.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date


# Prediction Endpoint Models

class PredictRequest(BaseModel):
    """What the user sends when asking for a forecast."""
    category: str = Field(..., description="Product category name in English")
    days: int = Field(default=30, ge=1, le=365, description="How many days to forecast")


class DailyForecast(BaseModel):
    """Forecast for a single day."""
    date: str
    predicted_orders: int
    predicted_lower: int
    predicted_upper: int


class PredictResponse(BaseModel):
    """The forecast response sent back to the user."""
    status: str = "success"
    category: str
    forecast_days: int
    total_forecast_orders: int
    avg_daily_forecast: float
    avg_historical_daily: float
    last_training_date: str
    forecast: List[DailyForecast]


class PredictError(BaseModel):
    """Error message if prediction fails."""
    status: str = "error"
    detail: str


# Optimization Endpoint Models

class CategoryCostInput(BaseModel):
    """Cost and stock info for one category."""
    category: str
    current_stock: float = Field(default=100, ge=0)
    unit_cost: float = Field(default=50.0, ge=0)
    selling_price: float = Field(default=80.0, ge=0)
    storage_per_unit: float = Field(default=1.0, ge=0)
    forecast_demand: Optional[float] = None
    forecast_std: Optional[float] = None


class OptimizeRequest(BaseModel):
    """Request to run inventory optimization."""
    categories: Optional[List[CategoryCostInput]] = Field(default=None)
    storage_capacity: Optional[float] = Field(default=None, ge=0)
    budget: Optional[float] = Field(default=None, ge=0)
    service_level: Optional[float] = Field(default=0.95, ge=0, le=1)


class InventoryRecommendation(BaseModel):
    """Optimization result for one category."""
    category: str
    current_stock: float
    forecast_demand: float
    reorder_quantity: float
    reorder_point: float
    safety_stock: float
    total_inventory_after_order: float
    unit_cost: float
    holding_cost: float


class OptimizeResponse(BaseModel):
    """Optimization results sent back to the user."""
    status: str = "success"
    total_cost: float
    recommendations: List[InventoryRecommendation]


class OptimizeError(BaseModel):
    """Error message if optimization fails."""
    status: str = "error"
    detail: str


# Health Check Model

class HealthResponse(BaseModel):
    """Shows API health and available models."""
    status: str = "healthy"
    models_loaded: int
    categories_available: List[str]
