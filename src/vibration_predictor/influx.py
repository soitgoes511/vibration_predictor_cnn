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

RUN_META_FIELDS = [
    "sample_rate_hz",
    "sample_count",
    "fft_size",
    "filter_cutoff_hz",
    "range_g",
    "send_time_domain",
    "window",
    "fw",
]


def _escape_flux(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_tag_filter(tag: str, values: tuple[str, ...] | list[str]) -> str:
    normalized = [v for v in values if str(v).strip()]
    if not normalized:
        return ""
    conditions = " or ".join([f'r["{tag}"] == "{_escape_flux(str(v))}"' for v in normalized])
    return f"  |> filter(fn: (r) => {conditions})\n"


def build_flux_query(cfg: InfluxConfig) -> str:
    operation_filter = ""
    if cfg.operation:
        operation_filter = f'  |> filter(fn: (r) => r["operation"] == "{_escape_flux(str(cfg.operation))}")\n'
    device_filter = _build_tag_filter("device_id", cfg.device_ids)

    stop_clause = f", stop: {cfg.stop}" if cfg.stop else ""
    measurement = _escape_flux(cfg.measurement)

    return (
        f'from(bucket: "{_escape_flux(cfg.bucket)}")\n'
        f"  |> range(start: {cfg.start}{stop_clause})\n"
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f"{operation_filter}"
        f"{device_filter}"
        '  |> filter(fn: (r) => r["_field"] == "frequencies" or r["_field"] == "x_freq" or r["_field"] == "y_freq" or r["_field"] == "z_freq")\n'
        '  |> pivot(rowKey: ["_time", "operation", "run_id"], columnKey: ["_field"], valueColumn: "_value")\n'
        '  |> keep(columns: ["_time", "operation", "run_id", "frequencies", "x_freq", "y_freq", "z_freq"])\n'
        '  |> sort(columns: ["operation", "run_id", "frequencies"])'
    )


def build_run_metadata_query(cfg: InfluxConfig) -> str:
    operation_filter = ""
    if cfg.operation:
        operation_filter = f'  |> filter(fn: (r) => r["operation"] == "{_escape_flux(str(cfg.operation))}")\n'
    device_filter = _build_tag_filter("device_id", cfg.device_ids)

    stop_clause = f", stop: {cfg.stop}" if cfg.stop else ""
    measurement = _escape_flux(cfg.metadata_measurement)
    field_filter = " or ".join([f'r["_field"] == "{field}"' for field in RUN_META_FIELDS])

    return (
        f'from(bucket: "{_escape_flux(cfg.bucket)}")\n'
        f"  |> range(start: {cfg.start}{stop_clause})\n"
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f"{operation_filter}"
        f"{device_filter}"
        f"  |> filter(fn: (r) => {field_filter})\n"
        '  |> pivot(rowKey: ["_time", "operation", "run_id", "device_id"], columnKey: ["_field"], valueColumn: "_value")\n'
        '  |> keep(columns: ["_time", "operation", "run_id", "device_id", "sample_rate_hz", "sample_count", "fft_size", "filter_cutoff_hz", "range_g", "send_time_domain", "window", "fw"])\n'
        '  |> sort(columns: ["operation", "run_id", "_time"])'
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


def _normalize_run_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["operation", "run_id", "device_id", "sample_rate_hz", "sample_count", "fft_size", "window", "fw"]
        )

    df = frame.copy()
    df = df.drop(
        columns=[c for c in ("result", "table", "_start", "_stop", "_measurement") if c in df.columns],
        errors="ignore",
    )

    required_base = {"operation", "run_id"}
    if not required_base.issubset(df.columns):
        missing = sorted(required_base.difference(set(df.columns)))
        raise ValueError(f"Run metadata query missing expected columns: {missing}")

    for col in ("operation", "run_id", "device_id", "window", "fw"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in ("sample_rate_hz", "sample_count", "fft_size", "filter_cutoff_hz", "range_g"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "_time" in df.columns:
        df = df.sort_values(["operation", "run_id", "_time"])
        df = df.groupby(["operation", "run_id"], as_index=False).tail(1)

    keep_cols = [c for c in ["operation", "run_id", "device_id", "sample_rate_hz", "sample_count", "fft_size", "window", "fw"] if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def _run_metadata_filters_enabled(cfg: InfluxConfig) -> bool:
    return (
        cfg.require_run_metadata
        or bool(cfg.device_ids)
        or cfg.expected_sample_rate_hz is not None
        or cfg.expected_fft_size is not None
        or bool(cfg.expected_window)
        or bool(cfg.expected_fw_prefix)
    )


def _apply_run_metadata_filters(freq_df: pd.DataFrame, meta_df: pd.DataFrame, cfg: InfluxConfig) -> pd.DataFrame:
    if meta_df.empty:
        if cfg.require_run_metadata:
            raise ValueError("Run metadata is required but no metadata rows were returned from InfluxDB")
        return freq_df

    valid = meta_df.copy()

    if cfg.device_ids and "device_id" in valid.columns:
        valid = valid[valid["device_id"].isin(cfg.device_ids)]

    if cfg.expected_sample_rate_hz is not None and "sample_rate_hz" in valid.columns:
        valid = valid[valid["sample_rate_hz"] == cfg.expected_sample_rate_hz]

    if cfg.expected_fft_size is not None and "fft_size" in valid.columns:
        valid = valid[valid["fft_size"] == cfg.expected_fft_size]

    if cfg.expected_window and "window" in valid.columns:
        valid = valid[valid["window"].astype(str).str.lower() == cfg.expected_window.lower()]

    if cfg.expected_fw_prefix and "fw" in valid.columns:
        valid = valid[valid["fw"].astype(str).str.startswith(cfg.expected_fw_prefix)]

    valid_keys = valid[["operation", "run_id"]].drop_duplicates()
    if valid_keys.empty:
        raise ValueError("No runs match configured metadata filters. Check influx.* metadata filter settings.")

    filtered = freq_df.merge(valid_keys, on=["operation", "run_id"], how="inner")
    if filtered.empty:
        raise ValueError("No frequency rows remain after run metadata filtering")

    return filtered.sort_values(["operation", "run_id", "frequencies"]).reset_index(drop=True)


def _query_to_dataframe(client: InfluxDBClient, query: str, org: str, empty_columns: list[str]) -> pd.DataFrame:
    raw = client.query_api().query_data_frame(query=query, org=org)
    if isinstance(raw, list):
        frames = [f for f in raw if isinstance(f, pd.DataFrame) and not f.empty]
        if not frames:
            return pd.DataFrame(columns=empty_columns)
        return pd.concat(frames, ignore_index=True)
    if isinstance(raw, pd.DataFrame):
        return raw if not raw.empty else pd.DataFrame(columns=empty_columns)
    return pd.DataFrame(columns=empty_columns)


def fetch_run_metadata_frame(cfg: InfluxConfig) -> pd.DataFrame:
    if not cfg.url or not cfg.org or not cfg.bucket:
        raise ValueError("Influx URL, org, and bucket are required")
    if not cfg.token:
        raise ValueError("Influx token is empty. Set INFLUXDB_TOKEN or influx.token in config")

    flux_query = build_run_metadata_query(cfg)
    with InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org, timeout=60_000) as client:
        raw_df = _query_to_dataframe(
            client=client,
            query=flux_query,
            org=cfg.org,
            empty_columns=["operation", "run_id", "device_id"],
        )

    return _normalize_run_metadata_frame(raw_df)


def fetch_frequency_frame(cfg: InfluxConfig) -> pd.DataFrame:
    if not cfg.url or not cfg.org or not cfg.bucket:
        raise ValueError("Influx URL, org, and bucket are required")
    if not cfg.token:
        raise ValueError("Influx token is empty. Set INFLUXDB_TOKEN or influx.token in config")

    flux_query = build_flux_query(cfg)
    with InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org, timeout=60_000) as client:
        merged = _query_to_dataframe(client=client, query=flux_query, org=cfg.org, empty_columns=REQUIRED_COLUMNS)

        freq_df = _normalize_query_frame(merged) if not merged.empty else pd.DataFrame(columns=REQUIRED_COLUMNS)
        if freq_df.empty:
            return freq_df

        if _run_metadata_filters_enabled(cfg):
            run_meta_query = build_run_metadata_query(cfg)
            raw_meta_df = _query_to_dataframe(
                client=client,
                query=run_meta_query,
                org=cfg.org,
                empty_columns=["operation", "run_id", "device_id"],
            )
            meta_df = _normalize_run_metadata_frame(raw_meta_df) if not raw_meta_df.empty else pd.DataFrame()
            freq_df = _apply_run_metadata_filters(freq_df, meta_df, cfg)

        return freq_df
