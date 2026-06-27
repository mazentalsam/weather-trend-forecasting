"""Tests for ML forecasting models."""

from __future__ import annotations

import numpy as np
import pytest
from src.models.ml import _build_model, fit_ml_models, tune_model


class TestBuildModel:
    @pytest.mark.parametrize("name", ["Ridge", "Random Forest", "XGBoost", "Gradient Boosting", "Neural Network (MLP)"])
    def test_builds_all_registered_models(self, name):
        model = _build_model(name, {})
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_config_override(self):
        model = _build_model("Ridge", {"alpha": 5.0})
        assert model.alpha == 5.0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            _build_model("NonexistentModel", {})


class TestFitMlModels:
    def test_returns_all_models(self, feature_matrix):
        train_X, train_y, test_X, test_y, test_idx = feature_matrix
        results = fit_ml_models(train_X, train_y, test_X, test_idx)
        assert len(results) == 5
        for name in ["Ridge", "Random Forest", "XGBoost", "Gradient Boosting", "Neural Network (MLP)"]:
            assert name in results

    def test_predictions_match_test_length(self, feature_matrix):
        train_X, train_y, test_X, test_y, test_idx = feature_matrix
        results = fit_ml_models(train_X, train_y, test_X, test_idx)
        for name, (_model, preds) in results.items():
            assert len(preds) == len(test_X), f"{name} prediction length mismatch"

    def test_predictions_are_finite(self, feature_matrix):
        train_X, train_y, test_X, test_y, test_idx = feature_matrix
        results = fit_ml_models(train_X, train_y, test_X, test_idx)
        for name, (_model, preds) in results.items():
            assert preds.notna().all(), f"{name} has NaN predictions"
            assert np.isfinite(preds).all(), f"{name} has infinite predictions"

    def test_predictions_have_correct_index(self, feature_matrix):
        train_X, train_y, test_X, test_y, test_idx = feature_matrix
        results = fit_ml_models(train_X, train_y, test_X, test_idx)
        for name, (_model, preds) in results.items():
            assert (preds.index == test_idx).all(), f"{name} index mismatch"


class TestTuneModel:
    def test_xgboost_tuning(self, feature_matrix):
        train_X, train_y, test_X, _, _ = feature_matrix
        grid = tune_model(
            "XGBoost",
            train_X.values, train_y.values,
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            n_splits=3,
        )
        assert hasattr(grid, "best_estimator_")
        assert hasattr(grid, "best_params_")
        preds = grid.best_estimator_.predict(test_X)
        assert len(preds) == len(test_X)

    def test_random_forest_tuning(self, feature_matrix):
        train_X, train_y, test_X, _, _ = feature_matrix
        grid = tune_model(
            "Random Forest",
            train_X.values, train_y.values,
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            n_splits=3,
        )
        assert hasattr(grid, "best_estimator_")

    def test_unsupported_model_raises(self, feature_matrix):
        train_X, train_y, _, _, _ = feature_matrix
        with pytest.raises(ValueError, match="Tuning not supported"):
            tune_model("Ridge", train_X.values, train_y.values, {})
