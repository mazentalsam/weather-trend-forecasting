"""Statistical time-series models — ARIMA, SARIMA, Holt-Winters, Prophet."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


def fit_arima(
    train: pd.Series,
    n_forecast: int,
    forecast_index: pd.DatetimeIndex,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.Series, str]:
    cfg = cfg or {}
    auto = auto_arima(
        train,
        seasonal=cfg.get("seasonal", False),
        stepwise=True,
        suppress_warnings=True,
        max_p=cfg.get("max_p", 5),
        max_q=cfg.get("max_q", 5),
        max_d=cfg.get("max_d", 2),
        information_criterion=cfg.get("information_criterion", "aic"),
    )
    order = auto.order
    label = f"ARIMA{order}"
    logger.info("Auto-selected %s (AIC: %.2f)", label, auto.aic())

    model = ARIMA(train, order=order).fit()
    pred = model.forecast(steps=n_forecast)
    pred.index = forecast_index
    return pred, label


def fit_sarima(
    train: pd.Series,
    n_forecast: int,
    forecast_index: pd.DatetimeIndex,
    order: tuple = (2, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> tuple[pd.Series, str]:
    label = f"SARIMA{order}x{seasonal_order}"
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    pred = fit.forecast(steps=n_forecast)
    pred.index = forecast_index
    return pred, label


def fit_holt_winters(
    train: pd.Series,
    n_forecast: int,
    forecast_index: pd.DatetimeIndex,
    seasonal_period: int = 7,
) -> tuple[pd.Series, str]:
    period = min(seasonal_period, len(train) // 3)
    if len(train) >= 2 * period:
        hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=period)
        label = "Holt-Winters"
    else:
        hw = ExponentialSmoothing(train, trend="add", seasonal=None)
        label = "Holt (No Seasonal)"
    fit = hw.fit(optimized=True)
    pred = fit.forecast(steps=n_forecast)
    pred.index = forecast_index
    return pred, label


def fit_prophet(
    train: pd.Series,
    n_forecast: int,
    forecast_index: pd.DatetimeIndex,
) -> tuple[pd.Series, str]:
    from prophet import Prophet

    prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=len(train) > 365,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=n_forecast)
    forecast = model.predict(future)
    pred = forecast.set_index("ds").loc[forecast_index, "yhat"]
    return pred, "Prophet"
