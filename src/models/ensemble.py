"""Ensemble methods — simple average and inverse-RMSE weighting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def build_ensemble(
    forecasts: dict[str, pd.Series],
    actual: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    common_idx = forecasts[next(iter(forecasts))].index
    for v in forecasts.values():
        common_idx = common_idx.intersection(v.index)

    actual_common = actual.loc[common_idx]

    simple = sum(v.loc[common_idx] for v in forecasts.values()) / len(forecasts)

    rmse_dict = {
        k: float(np.sqrt(mean_squared_error(actual_common, v.loc[common_idx])))
        for k, v in forecasts.items()
    }
    inv_rmse = {k: 1.0 / v for k, v in rmse_dict.items()}
    total = sum(inv_rmse.values())
    weights = {k: v / total for k, v in inv_rmse.items()}

    weighted = sum(weights[k] * v.loc[common_idx] for k, v in forecasts.items())

    logger.info(
        "Ensemble built from %d models — weights: %s",
        len(forecasts),
        {k: f"{v:.3f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])},
    )
    return simple, weighted, weights
