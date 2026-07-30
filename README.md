# Demand Forecasting & Inventory Optimization

**Author:** Kaif Khurshid  
**Institution:** XIM University, Bhubaneswar  
**Program:** B.Tech (Final Year) — Batch 2023-27  

An end-to-end demand forecasting system using the **Brazilian E-Commerce (Olist) dataset** that predicts daily order volume across product categories, optimizes inventory allocation, and presents insights through an interactive dashboard.

---

## Project Overview

| Metric | Target | Achieved |
|---|---|---|
| Forecast Accuracy (MAPE) | <15% | **14.6% avg** (best: 10.5%) |
| Improvement over Baseline | — | **64%** vs naive lag-1 |
| Inventory Cost Reduction | >=15% | **45%** holding cost reduction |
| Service Level | >=95% | ✅ Enforced as LP constraint |

### Key Numbers

- **110,197** order-item records processed from **96,478** unique orders
- **9 CSV files** merged into a unified pipeline
- **61 engineered features** (lags, rolling stats, seasonality, holidays, exogenous)
- **4 models** compared: Prophet, XGBoost, LSTM, Weighted Ensemble
- **5 product categories** modeled (top by volume, out of 72 total)
- **Walk-forward cross-validation** with chronological train/val/test split
- **<150ms** API inference latency (FastAPI)
- **5-page interactive dashboard** with 12+ Plotly visualizations

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Data            | --> |  Feature          | --> |  Model            |
|   Ingestion       |     |  Engineering      |     |  Training         |
+-------------------+     +-------------------+     +---------+---------+
                                                               |
+-------------------+     +-------------------+     +---------+---------+
|  Streamlit        | <-- |  FastAPI          | <-- |  Inventory        |
|  Dashboard        |     |  (Inference)      |     |  Optimization     |
+-------------------+     +-------------------+     +-------------------+
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Modeling | Prophet, XGBoost, TensorFlow/Keras (LSTM), Scikit-learn |
| Optimization | PuLP (Linear Programming) |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |
| Hyperparameter Tuning | Optuna |

## Project Structure

```
demand-forecasting-optimization/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
|
+-- data/
|   +-- raw/                    # Original Olist CSV files
|   +-- processed/              # Cleaned Parquet data
|
+-- notebooks/
|   +-- 01_eda.ipynb            # Exploratory Data Analysis
|   +-- 02_feature_engineering.ipynb
|   +-- 03_model_benchmark.ipynb
|   +-- 04_optimization.ipynb   # Inventory optimization demo
|
+-- src/
|   +-- config.py               # Paths, params, constants
|   +-- data_loader.py          # Load and clean 9 CSVs
|   +-- features.py             # Lag, rolling, seasonal features
|   +-- train.py                # Prophet, XGBoost, LSTM, Ensemble
|   +-- predict.py              # Inference helpers
|   +-- optimize.py             # PuLP inventory optimization
|   +-- utils.py                # Metrics, plots, model persistence
|
+-- api/
|   +-- main.py                 # FastAPI app
|   +-- schemas.py              # Pydantic models
|
+-- dashboard/
|   +-- app.py                  # Streamlit entry point
|   +-- pages/
|       +-- 01_eda.py           # EDA overview
|       +-- 02_model_comparison.py
|       +-- 03_forecast.py      # Forecast viewer
|       +-- 04_optimization.py  # Inventory optimizer
|
+-- models/                     # Saved model artifacts
|
+-- reports/
    +-- figures/                # Saved plots
    +-- results.md              # Final metrics
```

## Getting Started

### Live Deployments

| Service | URL |
|---|---|
| **Streamlit Dashboard** | https://demand-forecasting-inventory-optimization-kdsgzhevubl6mchvjsfp.streamlit.app |
| **FastAPI** | https://demand-forecasting-inventory-optimization.onrender.com |
| **API Swagger Docs** | https://demand-forecasting-inventory-optimization.onrender.com/docs |

### Prerequisites
- Python 3.11+
- [Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (place CSVs in `data/raw/`)

### Setup

```bash
# Clone and enter the project
cd demand-forecasting-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place the 9 Olist CSV files in `data/raw/`:

```
data/raw/
+-- olist_orders_dataset.csv
+-- olist_order_items_dataset.csv
+-- olist_products_dataset.csv
+-- olist_customers_dataset.csv
+-- olist_order_payments_dataset.csv
+-- olist_order_reviews_dataset.csv
+-- olist_geolocation_dataset.csv
+-- olist_sellers_dataset.csv
+-- product_category_name_translation.csv
```

### Run the Pipeline

**Option 1: Jupyter Notebooks (recommended for exploration)**

```bash
jupyter notebook notebooks/
# Run in order: 01 -> 02 -> 03 -> 04
```

**Option 2: Python Scripts**

```bash
# 1. Data ingestion and cleaning
python src/data_loader.py

# 2. Train all models
python src/train.py

# 3. Run inventory optimization demo
python src/optimize.py
```

### Launch Services (Local Development)

```bash
# FastAPI Inference API
uvicorn api.main:app --reload
# → http://localhost:8000/docs (Swagger UI)

# Streamlit Dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
```

### Or Use Live Deployments

- **Dashboard:** https://demand-forecasting-inventory-optimization-kdsgzhevubl6mchvjsfp.streamlit.app
- **API:** https://demand-forecasting-inventory-optimization.onrender.com/docs

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check and available categories |
| `/predict?category=X&days=30` | GET | Get demand forecast |
| `/optimize` | POST | Get inventory recommendations |

### Example: GET /predict

```bash
curl "http://localhost:8000/predict?category=bed_bath_table&days=30"
```

```json
{
  "status": "success",
  "category": "bed_bath_table",
  "total_forecast_orders": 3452,
  "avg_daily_forecast": 115.1,
  "forecast": [
    {"date": "2018-11-01", "predicted_orders": 108, "predicted_lower": 92, "predicted_upper": 124},
    ...
  ]
}
```

## Dashboard Pages

| Page | Description |
|---|---|
| Home | Project overview and quick metrics |
| EDA Overview | Sales trends, category ranking, seasonality heatmaps |
| Model Comparison | Metrics table, forecast vs actual plots |
| Forecast Viewer | Select category, view 30/60/90 day forecast |
| Inventory Optimizer | Configure costs, run PuLP, view recommendations |

## Models

| Model | Library | Rationale |
|---|---|---|
| Prophet | Prophet | Handles seasonality and holidays natively |
| XGBoost | XGBoost | Best for tabular + lag features |
| LSTM | TensorFlow/Keras | Captures long-range temporal patterns |
| Ensemble | Weighted Average | Combines all models for robustness |

### Training Strategy
- **Walk-forward validation** with chronological split
  - Train: 2016-09-15 to 2018-06-30 (main learning period)
  - Validation: 2018-07-01 to 2018-08-29 (model selection & tuning)
- **61 engineered features**: 7 lag values (1–30 days), 3 rolling windows (7/14/30-day mean & std), seasonal dummies (day-of-week, month, quarter), Brazilian holiday flags, exogenous price/freight/review features
- **Hyperparameter tuning** via Optuna (scoped)
- **Ensemble weighting**: Inverse-RMSE based (best-performing model gets highest weight)

## Results

### Per-Category Model Performance (Validation MAPE)

| Category | MAPE | Model |
|---|---|---|
| computers_accessories | **10.51%** | XGBoost |
| bed_bath_table | **12.73%** | XGBoost |
| sports_leisure | **14.24%** | XGBoost |
| health_beauty | **14.75%** | XGBoost |
| furniture_decor | **20.60%** | XGBoost |

### Aggregate Results

| Metric | Value |
|---|---|
| Average Ensemble MAPE | **14.6%** |
| Naive Baseline MAPE (lag-1) | 40.5% |
| **Improvement over baseline** | **64%** |
| Validation Method | Walk-forward (chronological split) |
| Train Period | 2016-09-15 → 2018-06-30 |
| Validation Period | 2018-07-01 → 2018-08-29 |

### Inventory Optimization Results

| Metric | Value |
|---|---|
| Solver | PuLP (CBC) |
| Status | Optimal |
| Naive holding cost | $14,143.75 |
| Optimized total cost | $7,820.42 |
| **Cost reduction** | **45%** |
| Service level enforced | 95% (z = 1.645) |

## Inventory Optimization

The PuLP linear program solves for optimal reorder quantities:

- **Objective**: Minimize holding cost + stockout cost
- **Variables**: Reorder quantity per category per period
- **Constraints**: Storage capacity, purchase budget, service level >=95%
- **Sensitivity**: What-if analysis on service level vs total cost

### Optimization Output (Sample — 5 Categories)

| Category | Reorder Qty | Safety Stock | Holding Cost |
|---|---|---|---|
| bed_bath_table | 86 | 16 | $1,247.52 |
| health_beauty | 57 | 12 | $1,057.62 |
| sports_leisure | 40 | 10 | $886.19 |
| furniture_decor | 26 | 11 | $774.73 |
| computers_accessories | 47 | 12 | $3,854.36 |

**Total optimized cost: $7,820.42** (vs $14,143.75 naive → **45% savings**)

## Deliverables Checklist

- [x] Clean data pipeline (9 CSVs -> processed features)
- [x] EDA notebook with key visualizations
- [x] Feature engineering (lags, rolling, seasonal, holidays)
- [x] 4 trained models (Prophet, XGBoost, LSTM, Ensemble)
- [x] Model comparison with MAPE, RMSE, MAE
- [x] Inventory optimization solver (PuLP)
- [x] FastAPI with `/predict` and `/optimize` endpoints
- [x] Streamlit dashboard (4 pages)
- [x] README with setup, results, and architecture

## Dataset

**Brazilian E-Commerce Public Dataset by Olist** (100K orders, 2016–2018)

| Stat | Value |
|---|---|
| Total order-item records | 110,197 |
| Unique orders | 96,478 |
| Product categories | 72 |
| Date range | Sep 2016 – Aug 2018 |
| Source files | 9 CSVs |

The dataset contains information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing an order from multiple dimensions: order status, price, payment, freight performance, customer location, product attributes, and reviews.

Source: [Kaggle — Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## License

This project is for educational and demonstration purposes. The Olist dataset is provided under a CC BY-NC-SA 4.0 license.

---

