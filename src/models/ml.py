"""Machine learning forecasting models with tuning support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type] = {
    "Ridge": Ridge,
    "Random Forest": RandomForestRegressor,
    "XGBoost": xgb.XGBRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Neural Network (MLP)": MLPRegressor,
}


def _build_model(name: str, cfg: dict[str, Any]) -> Any:
    defaults: dict[str, dict] = {
        "Ridge": {"alpha": cfg.get("alpha", 1.0)},
        "Random Forest": {
            "n_estimators": cfg.get("n_estimators", 200),
            "max_depth": cfg.get("max_depth", 10),
            "random_state": 42,
            "n_jobs": -1,
        },
        "XGBoost": {
            "n_estimators": cfg.get("n_estimators", 200),
            "max_depth": cfg.get("max_depth", 6),
            "learning_rate": cfg.get("learning_rate", 0.1),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        },
        "Gradient Boosting": {
            "n_estimators": cfg.get("n_estimators", 200),
            "max_depth": cfg.get("max_depth", 5),
            "learning_rate": cfg.get("learning_rate", 0.1),
            "random_state": 42,
        },
        "Neural Network (MLP)": {
            "hidden_layer_sizes": tuple(cfg.get("hidden_layer_sizes", [128, 64, 32])),
            "max_iter": cfg.get("max_iter", 500),
            "learning_rate_init": cfg.get("learning_rate_init", 0.001),
            "early_stopping": cfg.get("early_stopping", True),
            "random_state": 42,
            "validation_fraction": 0.1,
        },
    }
    return _MODEL_REGISTRY[name](**defaults[name])


def fit_ml_models(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    cfg: dict[str, Any] | None = None,
) -> dict[str, tuple[Any, pd.Series]]:
    cfg = cfg or {}
    results = {}
    for name in _MODEL_REGISTRY:
        model_cfg = cfg.get(name.lower().replace(" ", "_"), {})
        model = _build_model(name, model_cfg)
        model.fit(train_X, train_y)
        preds = pd.Series(model.predict(test_X), index=test_index)
        results[name] = (model, preds)
        logger.info("Trained %s — predictions shape: %s", name, preds.shape)
    return results


def tune_model(
    model_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    param_grid: dict[str, list],
    n_splits: int = 5,
) -> Any:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base_params: dict[str, Any] = {"random_state": 42}
    if model_name == "XGBoost":
        base = xgb.XGBRegressor(verbosity=0, n_jobs=-1, **base_params)
    elif model_name == "Random Forest":
        base = RandomForestRegressor(n_jobs=-1, **base_params)
    else:
        raise ValueError(f"Tuning not supported for {model_name}")

    grid = GridSearchCV(
        base,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(train_X, train_y)
    logger.info(
        "Tuning %s complete — best params: %s, best RMSE: %.4f",
        model_name, grid.best_params_, -grid.best_score_,
    )
    return grid


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)
