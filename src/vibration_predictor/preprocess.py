from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AXIS_COLUMNS = ("x_freq", "y_freq", "z_freq")


def load_labels_csv(path: str | Path) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    required = {"run_id", "label"}
    if not required.issubset(labels.columns):
        raise ValueError("Labels CSV must contain at least 'run_id' and 'label' columns")

    labels["run_id"] = labels["run_id"].astype(str).str.strip()
    labels["label"] = labels["label"].astype(str).str.strip()

    if "operation" in labels.columns:
        labels["operation"] = (
            labels["operation"]
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": "", "<NA>": ""})
        )

    labels = labels.replace({"": np.nan})
    labels = labels.dropna(subset=["run_id", "label"])

    if "operation" in labels.columns and labels["operation"].str.len().gt(0).all():
        labels = labels.drop_duplicates(subset=["operation", "run_id"], keep="last")
    else:
        labels = labels.drop(columns=["operation"], errors="ignore")
        labels = labels.drop_duplicates(subset=["run_id"], keep="last")

    labels = labels.reset_index(drop=True)
    return labels


def build_run_tensors(
    freq_df: pd.DataFrame,
    target_bins: int,
    min_bins_per_run: int,
    log_scale: bool,
    normalize_per_axis: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    if freq_df.empty:
        raise ValueError("No frequency data loaded from InfluxDB")

    tensors: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    grouped = freq_df.groupby(["operation", "run_id"], sort=False)
    for (operation, run_id), run in grouped:
        run_clean = run[["frequencies", *AXIS_COLUMNS]].copy()
        run_clean = run_clean.groupby("frequencies", as_index=False).mean(numeric_only=True)
        run_clean = run_clean.sort_values("frequencies")

        if len(run_clean) < min_bins_per_run:
            continue

        src_freq = run_clean["frequencies"].to_numpy(dtype=np.float32)
        if not np.isfinite(src_freq).all():
            continue

        freq_span = float(src_freq[-1] - src_freq[0])
        if freq_span <= 0.0:
            continue

        target_freq = np.linspace(src_freq[0], src_freq[-1], target_bins, dtype=np.float32)
        channels: list[np.ndarray] = []

        for axis_col in AXIS_COLUMNS:
            src_vals = run_clean[axis_col].to_numpy(dtype=np.float32)
            vals = np.interp(target_freq, src_freq, src_vals).astype(np.float32)
            vals = np.maximum(vals, 0.0)

            if log_scale:
                vals = np.log1p(vals)

            if normalize_per_axis:
                mean = float(vals.mean())
                std = float(vals.std())
                if std < 1e-6:
                    std = 1.0
                vals = (vals - mean) / std

            channels.append(vals.astype(np.float32))

        tensors.append(np.stack(channels, axis=0))
        metadata_rows.append(
            {
                "operation": str(operation),
                "run_id": str(run_id),
                "min_frequency_hz": float(src_freq[0]),
                "max_frequency_hz": float(src_freq[-1]),
                "source_bins": int(len(run_clean)),
            }
        )

    if not tensors:
        raise ValueError(
            "No usable runs after preprocessing. Check Influx data and dataset.min_bins_per_run setting."
        )

    tensor_array = np.stack(tensors, axis=0).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    return tensor_array, metadata


def attach_labels(metadata: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.copy()
    lbl = labels.copy()

    meta["run_id"] = meta["run_id"].astype(str)
    lbl["run_id"] = lbl["run_id"].astype(str)

    if "operation" in lbl.columns and lbl["operation"].astype(str).str.len().gt(0).all():
        lbl["operation"] = lbl["operation"].astype(str)
        merged = meta.merge(lbl[["operation", "run_id", "label"]], on=["operation", "run_id"], how="left")
    else:
        merged = meta.merge(lbl[["run_id", "label"]], on="run_id", how="left")

    return merged.reset_index(drop=True)
