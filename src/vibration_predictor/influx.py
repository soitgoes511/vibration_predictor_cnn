from __future__ import annotations

import pandas as pd
from influxdb_client import InfluxDBClient

from vibration_predictor.config import InfluxConfig


REQUIRED_COLUMNS = [
    "_time",
    "operation",
    "run_id",
    "frequencies",
    "x_freq",
    "y_freq",
    "z_freq",
]


def _escape_flux(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_flux_query(cfg: InfluxConfig) -> str:
    operation_filter = ""
    if cfg.operation:
        operation_filter = f'  |> filter(fn: (r) => r["operation"] == "{_escape_flux(str(cfg.operation))}")\n'

    stop_clause = f", stop: {cfg.stop}" if cfg.stop else ""
    measurement = _escape_flux(cfg.measurement)

    return (
        f'from(bucket: "{_escape_flux(cfg.bucket)}")\n'
        f"  |> range(start: {cfg.start}{stop_clause})\n"
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f"{operation_filter}"
        '  |> filter(fn: (r) => r["_field"] == "frequencies" or r["_field"] == "x_freq" or r["_field"] == "y_freq" or r["_field"] == "z_freq")\n'
        '  |> pivot(rowKey: ["_time", "operation", "run_id"], columnKey: ["_field"], valueColumn: "_value")\n'
        '  |> keep(columns: ["_time", "operation", "run_id", "frequencies", "x_freq", "y_freq", "z_freq"])\n'
        '  |> sort(columns: ["operation", "run_id", "frequencies"])'
    )


def _normalize_query_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df = df.drop(
        columns=[c for c in ("result", "table", "_start", "_stop", "_measurement") if c in df.columns],
        errors="ignore",
    )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Influx query missing expected columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["run_id"] = df["run_id"].astype(str)
    df["operation"] = df["operation"].astype(str)

    for col in ("frequencies", "x_freq", "y_freq", "z_freq"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frequencies", "x_freq", "y_freq", "z_freq"])
    df = df.sort_values(["operation", "run_id", "frequencies"]).reset_index(drop=True)
    return df


def fetch_frequency_frame(cfg: InfluxConfig) -> pd.DataFrame:
    if not cfg.url or not cfg.org or not cfg.bucket:
        raise ValueError("Influx URL, org, and bucket are required")
    if not cfg.token:
        raise ValueError("Influx token is empty. Set INFLUXDB_TOKEN or influx.token in config")

    flux_query = build_flux_query(cfg)
    with InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org, timeout=60_000) as client:
        raw = client.query_api().query_data_frame(query=flux_query, org=cfg.org)

    if isinstance(raw, list):
        frames = [f for f in raw if isinstance(f, pd.DataFrame) and not f.empty]
        if not frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        merged = pd.concat(frames, ignore_index=True)
    elif isinstance(raw, pd.DataFrame):
        if raw.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        merged = raw
    else:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return _normalize_query_frame(merged)
