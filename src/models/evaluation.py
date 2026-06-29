"""Model evaluation metrics and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastMetrics:
    model: str
    mae: float
    rmse: float
    mape: float
    r2: float

    def to_dict(self) -> dict:
        return {
            "Model": self.model,
            "Avg Error (°C)": round(self.mae, 4),
            "Error Size (RMSE)": round(self.rmse, 4),
            "Error % (MAPE)": round(self.mape, 4),
            "Accuracy (R²)": round(self.r2, 4),
        }


def evaluate_forecast(actual: pd.Series, predicted: pd.Series, model_name: str) -> ForecastMetrics:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100)
    r2 = r2_score(actual, predicted)
    return ForecastMetrics(model=model_name, mae=mae, rmse=rmse, mape=mape, r2=r2)


def compute_prediction_interval(
    actual: pd.Series,
    predicted: pd.Series,
    confidence: float = 0.95,
) -> tuple[pd.Series, pd.Series, float]:
    from scipy import stats as sp_stats

    residuals = actual - predicted
    residual_std = float(residuals.std())
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    upper = predicted + z * residual_std
    lower = predicted - z * residual_std
    return lower, upper, residual_std
