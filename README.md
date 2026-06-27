# Global Weather Trend Forecasting

End-to-end weather analysis and forecasting pipeline that ingests global weather data from 190+ countries, runs 13 forecasting models (statistical + ML + neural network + ensemble), validates data quality, performs SHAP-based explainability, tracks experiments, and serves results through an interactive 9-page Streamlit dashboard.

[![CI](https://github.com/mazentalsam/weather-trend-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/mazentalsam/weather-trend-forecasting/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B.svg)](https://streamlit.io)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Results at a Glance

| Model | Type | RMSE | R² |
|-------|------|------|----|
| Weighted Ensemble | Inverse-RMSE weighted avg of top ML models | **Best** | **Best** |
| XGBoost (Tuned) | GridSearchCV + TimeSeriesSplit | Top-3 | Top-3 |
| Random Forest (Tuned) | GridSearchCV + TimeSeriesSplit | Top-3 | Top-3 |
| Neural Network (MLP) | 3-layer MLP with early stopping | Top-5 | Top-5 |
| Prophet | Additive decomposition | Mid | Mid |
| SARIMA | Seasonal ARIMA(2,1,1)(1,1,1,7) | Mid | Mid |
| ARIMA (auto) | Order selected via AIC minimization | Baseline | Baseline |

> Exact metrics depend on the dataset snapshot. Run the dashboard to see live numbers, or check `experiments.csv` for logged run history.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **`TimeSeriesSplit` for CV** | Standard k-fold leaks future data into training — `TimeSeriesSplit` preserves temporal order, which is critical for time-series evaluation |
| **Winsorization over removal** | Dropping outliers loses valid extreme weather events; clipping at 1st/99th percentile preserves sample size while limiting influence |
| **Inverse-RMSE ensemble weighting** | Better-performing models get proportionally more weight — more robust than equal averaging and avoids overfitting to a single model |
| **Residual-based prediction intervals** | Uses actual residual distribution rather than naive ±RMSE, giving calibrated 95% confidence intervals |
| **SHAP explainability** | SHAP TreeExplainer provides game-theoretic feature attribution — shows both global importance and per-sample prediction breakdowns |
| **4-method feature importance** | RF MDI + permutation + correlation + SHAP cross-validated against each other — features ranking high across all 4 are reliably important |
| **ADF stationarity testing** | Augmented Dickey-Fuller test before ARIMA modeling verifies time-series stationarity assumptions |
| **Cyclical encoding (sin/cos)** | Day-of-year as a linear feature creates an artificial discontinuity at year boundaries — sin/cos preserves the circular nature of seasonal patterns |
| **YAML-driven configuration** | All hyperparameters are centralized in `config/default.yaml` — no magic numbers scattered across source files |
| **Dynamic column identification** | Column names are detected via pattern matching rather than hardcoded — the pipeline adapts to schema changes without code modifications |

---

## Architecture

```
weather-trend-forecasting/
│
├── app/
│   └── streamlit_app.py          # Interactive multi-page dashboard
│
├── src/                           # Installable Python package
│   ├── config.py                  # YAML config loader
│   ├── data/
│   │   ├── loader.py              # Kaggle download + column identification
│   │   ├── cleaning.py            # Imputation, outlier handling, normalization
│   │   └── validation.py          # Schema validation + data quality reporting
│   ├── features/
│   │   └── engineering.py         # Lag features, rolling stats, cyclical encoding
│   ├── models/
│   │   ├── statistical.py         # ARIMA, SARIMA, Holt-Winters, Prophet
│   │   ├── ml.py                  # Ridge, Random Forest, XGBoost, Gradient Boosting, MLP
│   │   ├── ensemble.py            # Simple avg + inverse-RMSE weighted ensemble
│   │   ├── evaluation.py          # MAE, RMSE, MAPE, R², prediction intervals
│   │   └── feature_importance.py  # RF MDI, permutation, correlation, SHAP
│   ├── analysis/
│   │   ├── anomaly.py             # Isolation Forest + Z-score detection
│   │   ├── climate.py             # Climate zones, hemisphere patterns
│   │   ├── air_quality.py         # AQ-weather correlation analysis
│   │   └── spatial.py             # Continent mapping via pycountry_convert
│   ├── visualization/
│   │   └── plots.py               # Reusable Plotly chart library
│   └── tracking/
│       └── experiment.py          # Lightweight experiment logger (CSV-backed)
│
├── config/
│   └── default.yaml               # All hyperparameters and thresholds
│
├── tests/                         # pytest suite (60+ tests across all modules)
├── notebooks/
│   └── exploration.ipynb          # Full narrative EDA notebook
│
├── .github/workflows/ci.yml      # GitHub Actions: lint + type-check + test + coverage
├── Dockerfile                     # Container for one-command deployment
├── docker-compose.yml
├── Makefile                       # install / dev / test / coverage / lint / run / docker
├── pyproject.toml                 # Build config, ruff, mypy, pytest settings
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/mazentalsam/weather-trend-forecasting.git
cd weather-trend-forecasting
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"                           # editable install with dev tools
```

### 2. Configure Kaggle API

The dataset downloads automatically via `kagglehub`. Set up your credentials:

```bash
# Download kaggle.json from https://www.kaggle.com/settings
# Place at ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<you>\.kaggle\kaggle.json (Windows)
```

### 3. Launch the Dashboard

```bash
make run
# or: streamlit run app/streamlit_app.py
```

### 4. Run with Docker (alternative)

```bash
docker compose up --build
# Dashboard available at http://localhost:8501
```

---

## What the Dashboard Does

| Page | What You See |
|------|-------------|
| **Overview** | Dataset metrics, data quality report, feature distributions, PM Accelerator mission |
| **Exploratory Analysis** | Temperature trends, correlation heatmaps, country rankings, monthly patterns |
| **Anomaly Detection** | Interactive Isolation Forest with adjustable contamination, Z-score extremes |
| **Forecasting** | ADF stationarity test, one-click training of all 13 models, live RMSE comparison, forecast overlay with 95% prediction intervals, residual diagnostics |
| **Feature Importance** | 4-method comparison: RF MDI, Permutation, SHAP (beeswarm + waterfall), Correlation — with cross-method ranking table |
| **Climate Analysis** | Climate zone temperature curves, hemisphere seasonal inversion, variability boxplots |
| **Air Quality** | AQ-weather correlation matrix, strong-correlation table, AQ by climate zone |
| **Spatial Analysis** | Interactive globe map, continent comparison, latitude-temperature gradient |
| **Key Insights** | Auto-generated report with temperature patterns, geographic findings, anomaly summary, model leaderboard, and key takeaways |

---

## Models (13 total)

| # | Model | Category | Key Detail |
|---|-------|----------|------------|
| 1 | ARIMA | Statistical | Order auto-selected via `pmdarima` (AIC minimization) |
| 2 | SARIMA | Statistical | Weekly seasonality (2,1,1)(1,1,1,7) |
| 3 | Holt-Winters | Exponential Smoothing | Additive trend + seasonal |
| 4 | Prophet | Additive | Facebook's decomposable time-series model |
| 5 | Ridge Regression | Linear ML | L2-regularized baseline |
| 6 | Random Forest | Tree Ensemble | 200 estimators, max_depth=10 |
| 7 | XGBoost | Gradient Boosting | 200 rounds, lr=0.1 |
| 8 | Gradient Boosting | Gradient Boosting | sklearn implementation |
| 9 | Neural Network (MLP) | Deep Learning | 3-layer MLP (128→64→32) with early stopping |
| 10 | XGBoost (Tuned) | Optimized | GridSearchCV + 5-fold TimeSeriesSplit |
| 11 | Random Forest (Tuned) | Optimized | GridSearchCV + 5-fold TimeSeriesSplit |
| 12 | Ensemble (Simple) | Averaging | Mean of top ML models |
| 13 | Ensemble (Weighted) | Averaging | Inverse-RMSE weighted combination |

---

## Methodology

- **Data validation**: Schema checks, range validation, missing-value analysis, duplicate detection — runs before any transformations
- **Data cleaning**: Median/mode imputation, IQR outlier detection, 1st/99th percentile winsorization, StandardScaler normalization
- **Feature engineering**: 14 lag features, rolling statistics (mean/std/min/max for 7 & 14-day windows), cyclical day-of-year encoding (sin/cos), temporal features
- **Stationarity testing**: Augmented Dickey-Fuller test before ARIMA modeling — verifies assumptions and informs differencing order
- **Cross-validation**: 5-fold `TimeSeriesSplit` — respects temporal ordering, prevents data leakage
- **Hyperparameter tuning**: `GridSearchCV` over XGBoost and Random Forest parameter grids
- **Feature importance**: 4 methods (RF MDI, permutation, correlation, SHAP) — cross-method ranking identifies reliably important features
- **SHAP explainability**: TreeExplainer for global importance (beeswarm) + per-sample waterfall explanations
- **Anomaly detection**: Isolation Forest (configurable contamination) + Z-score (threshold: 3)
- **Prediction intervals**: Residual-based 95% confidence intervals (not naive ±RMSE)
- **Residual diagnostics**: Distribution, Q-Q plot, temporal pattern, ACF, Shapiro-Wilk normality test
- **Spatial mapping**: `pycountry_convert` for accurate country-to-continent assignment
- **Experiment tracking**: Every model run is logged with metrics, hyperparameters, and timestamps

---

## Development

```bash
make dev         # Editable install with dev dependencies
make test        # Run pytest suite
make coverage    # Run tests with coverage report (HTML output in htmlcov/)
make lint        # Ruff linter
make format      # Ruff formatter
make typecheck   # mypy type checking
make clean       # Remove caches and artifacts
```

All config lives in [`config/default.yaml`](config/default.yaml) — hyperparameters, thresholds, and model settings are centralized there, not scattered across code.

### Testing

The test suite covers all modules with 60+ tests:

| Module | Tests Cover |
|--------|------------|
| `data/cleaning` | Imputation strategies, outlier detection, winsorization, normalization, datetime parsing |
| `data/validation` | Schema checks, missing-value detection, range validation, duplicate detection |
| `features/engineering` | Lag feature correctness, rolling statistics, cyclical encoding |
| `models/statistical` | ARIMA, SARIMA, Holt-Winters, Prophet — forecast shape, index alignment, config handling |
| `models/ml` | All 5 ML models (incl. MLP) — training, prediction shape, finite outputs, hyperparameter tuning |
| `models/feature_importance` | RF MDI, permutation, correlation, SHAP — ranking correctness, value ranges, edge cases |
| `models/ensemble` | Weight normalization, better-model-gets-higher-weight, index intersection |
| `models/evaluation` | Metric computation, prediction intervals, perfect-prediction edge case |
| `analysis/anomaly` | Isolation Forest contamination control, Z-score thresholds, NaN handling |
| `analysis/climate` | Climate zone classification across all latitude bands, hemisphere assignment |
| `analysis/spatial` | Country-to-continent mapping for all 6 continents, unknown-country handling |
| `analysis/air_quality` | Correlation computation, value ranges, missing-column handling |
| `tracking/experiment` | Log/load roundtrip, append behavior, best-run queries |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | pandas, numpy, kagglehub |
| **Validation** | Custom schema validation with range checks and quality reporting |
| **ML / Stats** | scikit-learn, statsmodels, XGBoost, Prophet, pmdarima, SHAP |
| **Visualization** | Plotly, matplotlib, seaborn, Folium |
| **Dashboard** | Streamlit |
| **Spatial** | pycountry_convert, Folium |
| **Tracking** | Custom CSV-backed experiment logger |
| **CI/CD** | GitHub Actions (ruff + mypy + pytest + coverage) |
| **Deployment** | Docker, docker-compose |
| **Config** | YAML (PyYAML) |

---

## Dataset

[Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) — daily weather observations from 190+ countries covering temperature, precipitation, humidity, wind, air quality, and more.

---

## License

MIT
