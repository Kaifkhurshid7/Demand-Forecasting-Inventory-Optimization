# 📊 Demand Forecasting & Inventory Optimization

An end-to-end demand forecasting system using the **Brazilian E-Commerce (Olist) dataset** that predicts daily order volume across product categories, optimizes inventory allocation, and presents insights through an interactive dashboard.

---

## 🎯 Project Overview

| **Metric** | **Target** |
|---|---|
| Forecast Accuracy (MAPE) | <15% |
| RMSE | Minimize |
| Inventory Cost Reduction | ≥15% estimated |
| Service Level | ≥95% |

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data       │ ──► │  Feature     │ ──► │  Model       │
│   Ingestion  │     │  Engineering │     │  Training    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Streamlit   │ ◄── │  FastAPI     │ ◄── │  Inventory   │
│  Dashboard   │     │  (Inference) │     │  Optimization│
└──────────────┘     └──────────────┘     └──────────────┘
```

## 🧰 Tech Stack

| **Layer** | **Technology** |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Modeling | Prophet, XGBoost, TensorFlow/Keras (LSTM), Scikit-learn |
| Optimization | PuLP (Linear Programming) |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |
| Experiment Tracking | Optuna (hyperparameter tuning) |

## 📁 Project Structure

```
demand-forecasting-optimization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Original Olist CSV files
│   └── processed/              # Cleaned Parquet data
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_benchmark.ipynb
│   └── 04_optimization.ipynb   # Inventory optimization demo
│
├── src/
│   ├── config.py               # Paths, params, constants
│   ├── data_loader.py          # Load & clean 9 CSVs
│   ├── features.py             # Lag, rolling, seasonal features
│   ├── train.py                # Prophet, XGBoost, LSTM, Ensemble
│   ├── predict.py              # Inference helpers
│   ├── optimize.py             # PuLP inventory optimization
│   └── utils.py                # Metrics, plots, model persistence
│
├── api/
│   ├── main.py                 # FastAPI app
│   └── schemas.py              # Pydantic models
│
├── dashboard/
│   ├── app.py                  # Streamlit entry point
│   └── pages/
│       ├── 01_eda.py           # EDA overview
│       ├── 02_model_comparison.py
│       ├── 03_forecast.py      # Forecast viewer
│       └── 04_optimization.py  # Inventory optimizer
│
├── models/                     # Saved model artifacts
│
└── reports/
    ├── figures/                # Saved plots
    └── results.md              # Final metrics
```

## 🚀 Getting Started

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
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_products_dataset.csv
├── olist_customers_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

### Run the Pipeline

**Option 1: Jupyter Notebooks (recommended for exploration)**

```bash
jupyter notebook notebooks/
# Run in order: 01 → 02 → 03 → 04
```

**Option 2: Python Scripts**

```bash
# 1. Data ingestion & cleaning
python src/data_loader.py

# 2. Train all models
python src/train.py

# 3. Run inventory optimization demo
python src/optimize.py
```

### Launch Services

```bash
# FastAPI Inference API
uvicorn api.main:app --reload
# → http://localhost:8000/docs (Swagger UI)

# Streamlit Dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
```

## 📡 API Endpoints

| **Endpoint** | **Method** | **Description** |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check & available categories |
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

## 📊 Dashboard Pages

| **Page** | **Description** |
|---|---|
| **Home** | Project overview and quick metrics |
| **EDA Overview** | Sales trends, category ranking, seasonality heatmaps |
| **Model Comparison** | Metrics table, forecast vs actual plots |
| **Forecast Viewer** | Select category, view 30/60/90 day forecast |
| **Inventory Optimizer** | Configure costs, run PuLP, view recommendations |

## 🧠 Models

| **Model** | **Library** | **Why** |
|---|---|---|
| Prophet | Prophet | Handles seasonality & holidays natively |
| XGBoost | XGBoost | Best for tabular + lag features |
| LSTM | TensorFlow/Keras | Captures long-range temporal patterns |
| Ensemble | Weighted Average | Combines all models for robustness |

### Training Strategy
- **Walk-forward validation** with chronological split
  - Train: 2016-09 to 2018-06
  - Validation: 2018-07 to 2018-08
  - Test: 2018-09 to 2018-10
- **Features**: 30-day lags, 7/14/30-day rolling stats, seasonal dummies, holiday flags
- **Hyperparameter tuning** via Optuna (in scoping)

## 📈 Results

| **Metric** | **Target** | **Status** |
|---|---|---|
| MAPE | <15% | ✅ Achievable with ensemble |
| RMSE | Minimize | ✅ Tracked per category |
| Service Level | ≥95% | ✅ Constraint in optimizer |

*Detailed results in [`reports/results.md`](reports/results.md)*

## 🧪 Inventory Optimization

The PuLP linear program solves for optimal reorder quantities:

- **Objective**: Minimize holding cost + stockout cost
- **Variables**: Reorder quantity per category per period
- **Constraints**: Storage capacity, purchase budget, service level ≥95%
- **Sensitivity**: What-if analysis on service level vs total cost

## 📦 Deliverables Checklist

- [x] Clean data pipeline (9 CSVs → processed features)
- [x] EDA notebook with key visualizations
- [x] Feature engineering (lags, rolling, seasonal, holidays)
- [x] 4 trained models (Prophet, XGBoost, LSTM, Ensemble)
- [x] Model comparison with MAPE, RMSE, MAE
- [x] Inventory optimization solver (PuLP)
- [x] FastAPI with `/predict` and `/optimize` endpoints
- [x] Streamlit dashboard (4 pages)
- [x] README with setup, results, and architecture

## 📚 Dataset

**Brazilian E-Commerce Public Dataset by Olist** (100K orders, 2016–2018)

The dataset contains information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing an order from multiple dimensions: order status, price, payment, freight performance, customer location, product attributes, and reviews.

Source: [Kaggle — Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## 📝 License

This project is for educational and demonstration purposes. The Olist dataset is provided under a CC BY-NC-SA 4.0 license.
