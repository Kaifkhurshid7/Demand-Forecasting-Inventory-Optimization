"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date


# ──────────────────────────────────────────────
# Prediction Schemas
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request schema for GET /predict."""
    category: str = Field(..., description="Product category name (English)")
    days: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")


class DailyForecast(BaseModel):
    """A single day's forecast."""
    date: str
    predicted_orders: int
    predicted_lower: int
    predicted_upper: int


class PredictResponse(BaseModel):
    """Response schema for GET /predict."""
    status: str = "success"
    category: str
    forecast_days: int
    total_forecast_orders: int
    avg_daily_forecast: float
    avg_historical_daily: float
    last_training_date: str
    forecast: List[DailyForecast]


class PredictError(BaseModel):
    """Error response for prediction failures."""
    status: str = "error"
    detail: str


# ──────────────────────────────────────────────
# Optimization Schemas
# ──────────────────────────────────────────────

class CategoryCostInput(BaseModel):
    """Cost and stock data for a single category."""
    category: str
    current_stock: float = Field(default=100, ge=0)
    unit_cost: float = Field(default=50.0, ge=0)
    selling_price: float = Field(default=80.0, ge=0)
    storage_per_unit: float = Field(default=1.0, ge=0)
    forecast_demand: Optional[float] = None  # Use predicted if not provided
    forecast_std: Optional[float] = None


class OptimizeRequest(BaseModel):
    """Request schema for POST /optimize."""
    categories: Optional[List[CategoryCostInput]] = Field(default=None, description="Custom cost data per category")
    storage_capacity: Optional[float] = Field(default=None, ge=0)
    budget: Optional[float] = Field(default=None, ge=0)
    service_level: Optional[float] = Field(default=0.95, ge=0, le=1)


class InventoryRecommendation(BaseModel):
    """Single category recommendation."""
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
    """Response schema for POST /optimize."""
    status: str = "success"
    total_cost: float
    recommendations: List[InventoryRecommendation]


class OptimizeError(BaseModel):
    """Error response for optimization failures."""
    status: str = "error"
    detail: str


# ──────────────────────────────────────────────
# Health / Info
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: int
    categories_available: List[str]
