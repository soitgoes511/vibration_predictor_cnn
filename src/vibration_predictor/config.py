from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InfluxConfig:
    url: str
    org: str
    bucket: str
    token: str
    measurement: str
    operation: str | None
    start: str
    stop: str | None


@dataclass(frozen=True)
class DatasetConfig:
    target_bins: int
    min_bins_per_run: int
    log_scale: bool
    normalize_per_axis: bool


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_split: float
    val_split: float
    early_stopping_patience: int


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class PathsConfig:
    labels_csv: str
    output_dir: str


@dataclass(frozen=True)
class ExperimentConfig:
    influx: InfluxConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    paths: PathsConfig

    @property
    def labels_path(self) -> Path:
        return Path(self.paths.labels_csv)

    @property
    def output_path(self) -> Path:
        return Path(self.paths.output_dir)


def _as_tuple_int(raw: Any, field_name: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return fallback
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    raise ValueError(f"{field_name} must be a list or tuple of integers")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    influx_raw = raw.get("influx", {})
    token = os.getenv("INFLUXDB_TOKEN", influx_raw.get("token", ""))
    influx = InfluxConfig(
        url=str(influx_raw.get("url", "http://127.0.0.1:8086")),
        org=str(influx_raw.get("org", "")),
        bucket=str(influx_raw.get("bucket", "")),
        token=str(token),
        measurement=str(influx_raw.get("measurement", "accelfreq")),
        operation=influx_raw.get("operation"),
        start=str(influx_raw.get("start", "-30d")),
        stop=influx_raw.get("stop"),
    )

    dataset_raw = raw.get("dataset", {})
    dataset = DatasetConfig(
        target_bins=int(dataset_raw.get("target_bins", 1024)),
        min_bins_per_run=int(dataset_raw.get("min_bins_per_run", 128)),
        log_scale=bool(dataset_raw.get("log_scale", True)),
        normalize_per_axis=bool(dataset_raw.get("normalize_per_axis", True)),
    )

    training_raw = raw.get("training", {})
    training = TrainingConfig(
        random_seed=int(training_raw.get("random_seed", 42)),
        batch_size=int(training_raw.get("batch_size", 32)),
        epochs=int(training_raw.get("epochs", 60)),
        learning_rate=float(training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(training_raw.get("weight_decay", 1e-4)),
        train_split=float(training_raw.get("train_split", 0.7)),
        val_split=float(training_raw.get("val_split", 0.15)),
        early_stopping_patience=int(training_raw.get("early_stopping_patience", 10)),
    )

    if training.train_split <= 0 or training.val_split <= 0:
        raise ValueError("train_split and val_split must be positive")
    if training.train_split + training.val_split >= 1.0:
        raise ValueError("train_split + val_split must be less than 1.0")

    model_raw = raw.get("model", {})
    conv_channels = _as_tuple_int(model_raw.get("conv_channels"), "model.conv_channels", (32, 64, 128))
    kernel_sizes = _as_tuple_int(model_raw.get("kernel_sizes"), "model.kernel_sizes", (7, 5, 3))
    model = ModelConfig(
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dropout=float(model_raw.get("dropout", 0.3)),
    )

    if len(model.conv_channels) != len(model.kernel_sizes):
        raise ValueError("model.conv_channels and model.kernel_sizes must have the same length")

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        labels_csv=str(paths_raw.get("labels_csv", "data/labels.csv")),
        output_dir=str(paths_raw.get("output_dir", "artifacts/default_run")),
    )

    return ExperimentConfig(
        influx=influx,
        dataset=dataset,
        training=training,
        model=model,
        paths=paths,
    )
