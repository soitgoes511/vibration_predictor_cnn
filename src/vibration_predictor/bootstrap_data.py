from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _subsample_per_class(
    X: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    max_samples_per_class: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if max_samples_per_class <= 0:
        return X, labels, meta

    rng = np.random.default_rng(random_seed)
    keep_indices: list[int] = []
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        if len(label_idx) <= max_samples_per_class:
            keep_indices.extend(label_idx.tolist())
            continue
        chosen = rng.choice(label_idx, size=max_samples_per_class, replace=False)
        keep_indices.extend(chosen.tolist())

    keep = np.sort(np.asarray(keep_indices, dtype=np.int64))
    return X[keep], labels[keep], meta.iloc[keep].reset_index(drop=True)


def load_bootstrap_npz(
    path: str | Path,
    max_samples_per_class: int | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Bootstrap dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "labels" not in data:
        raise ValueError("Bootstrap NPZ must contain arrays 'X' and 'labels'")

    X = np.asarray(data["X"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(str)

    if X.ndim != 3 or X.shape[1] != 3:
        raise ValueError("Bootstrap X must have shape [N, 3, num_bins]")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("Bootstrap X and labels length mismatch")

    run_id = np.asarray(data["run_id"]).astype(str) if "run_id" in data else np.array([f"bootstrap_{i}" for i in range(len(X))], dtype=object)
    operation = np.asarray(data["operation"]).astype(str) if "operation" in data else np.array(["BOOTSTRAP"] * len(X), dtype=object)
    device_id = np.asarray(data["device_id"]).astype(str) if "device_id" in data else np.array(["BOOTSTRAP"] * len(X), dtype=object)
    source = np.asarray(data["source"]).astype(str) if "source" in data else np.array(["bootstrap"] * len(X), dtype=object)

    meta = pd.DataFrame(
        {
            "operation": operation,
            "run_id": run_id,
            "device_id": device_id,
            "label": labels,
            "source": source,
        }
    )

    if max_samples_per_class is not None:
        X, labels, meta = _subsample_per_class(
            X=X,
            labels=labels,
            meta=meta,
            max_samples_per_class=max_samples_per_class,
            random_seed=random_seed,
        )

    return X, labels, meta.reset_index(drop=True)
