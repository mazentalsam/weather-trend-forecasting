"""Lightweight experiment tracker — logs model runs to CSV."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "experiments.csv"


@dataclass
class ExperimentRun:
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    dataset_size: int = 0
    train_size: int = 0
    test_size: int = 0
    hyperparameters: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        row = asdict(self)
        row["hyperparameters"] = json.dumps(row["hyperparameters"])
        return row


class ExperimentTracker:
    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH

    def log(self, run: ExperimentRun) -> None:
        row = run.to_row()
        df = pd.DataFrame([row])

        if self.log_path.exists():
            df.to_csv(self.log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.log_path, index=False)

        logger.info("Logged experiment: %s (RMSE=%.4f)", run.model_name, run.rmse)

    def load(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.log_path)
        df["hyperparameters"] = df["hyperparameters"].apply(json.loads)
        return df

    def best_run(self, metric: str = "rmse") -> pd.Series | None:
        df = self.load()
        if df.empty:
            return None
        ascending = metric != "r2"
        return df.sort_values(metric, ascending=ascending).iloc[0]
