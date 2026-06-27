"""Feature importance analysis — RF, Permutation, Correlation, SHAP."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_rf_importance(
    model, feature_names: list[str]
) -> pd.Series:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)


def compute_perm_importance(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
) -> pd.Series:
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    imp = pd.Series(result.importances_mean, index=feature_names)
    return imp.sort_values(ascending=False)


def compute_correlation_importance(
    df: pd.DataFrame, target_col: str
) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        return pd.Series(dtype=float)
    corr = numeric.corr()[target_col].drop(target_col, errors="ignore").abs()
    return corr.sort_values(ascending=False)


def compute_shap_values(
    model, X: pd.DataFrame, max_samples: int = 500
) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    if not HAS_SHAP:
        raise ImportError("shap is required for SHAP analysis — pip install shap")

    sample = X.sample(n=min(max_samples, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
    logger.info("SHAP values computed for %d samples, %d features", len(sample), len(X.columns))
    return mean_abs.sort_values(ascending=False), shap_vals, sample
