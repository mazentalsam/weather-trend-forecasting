"""Tests for experiment tracking module."""

from __future__ import annotations

import json

import pytest
from src.tracking.experiment import ExperimentRun, ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(log_path=tmp_path / "test_experiments.csv")


@pytest.fixture
def sample_run():
    return ExperimentRun(
        model_name="TestModel",
        mae=1.5, rmse=2.0, mape=10.0, r2=0.85,
        dataset_size=1000, train_size=800, test_size=200,
        hyperparameters={"n_estimators": 100, "max_depth": 5},
    )


class TestExperimentRun:
    def test_to_row(self, sample_run):
        row = sample_run.to_row()
        assert row["model_name"] == "TestModel"
        assert row["rmse"] == 2.0
        assert json.loads(row["hyperparameters"])["n_estimators"] == 100

    def test_default_timestamp(self, sample_run):
        assert sample_run.timestamp is not None
        assert "T" in sample_run.timestamp


class TestExperimentTracker:
    def test_log_creates_file(self, tracker, sample_run):
        tracker.log(sample_run)
        assert tracker.log_path.exists()

    def test_log_and_load_roundtrip(self, tracker, sample_run):
        tracker.log(sample_run)
        df = tracker.load()
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "TestModel"
        assert df.iloc[0]["rmse"] == 2.0

    def test_multiple_logs_append(self, tracker, sample_run):
        tracker.log(sample_run)
        run2 = ExperimentRun(model_name="Model2", mae=1.0, rmse=1.5, mape=8.0, r2=0.90)
        tracker.log(run2)
        df = tracker.load()
        assert len(df) == 2

    def test_best_run(self, tracker):
        tracker.log(ExperimentRun(model_name="Bad", mae=5.0, rmse=6.0, mape=30.0, r2=0.5))
        tracker.log(ExperimentRun(model_name="Good", mae=1.0, rmse=1.2, mape=5.0, r2=0.95))
        best = tracker.best_run("rmse")
        assert best["model_name"] == "Good"

    def test_best_run_r2_descending(self, tracker):
        tracker.log(ExperimentRun(model_name="Low", mae=5.0, rmse=6.0, mape=30.0, r2=0.5))
        tracker.log(ExperimentRun(model_name="High", mae=1.0, rmse=1.2, mape=5.0, r2=0.95))
        best = tracker.best_run("r2")
        assert best["model_name"] == "High"

    def test_load_empty(self, tracker):
        df = tracker.load()
        assert df.empty

    def test_best_run_empty(self, tracker):
        assert tracker.best_run() is None
