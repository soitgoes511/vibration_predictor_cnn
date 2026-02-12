from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _guess_label_from_name(name: str) -> str | None:
    low = name.lower()
    if any(token in low for token in ("normal", "baseline", "healthy")):
        return "healthy"
    if "inner" in low or re.search(r"(^|[_-])ir([_-]|$)", low):
        return "inner_race_fault"
    if "outer" in low or re.search(r"(^|[_-])or([_-]|$)", low):
        return "outer_race_fault"
    if "ball" in low or re.search(r"(^|[_-])b([_-]|$)", low):
        return "ball_fault"
    return None


def _load_label_map(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["pattern", "label"])
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Label map CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"pattern", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Label map CSV must contain columns: pattern,label")
    df = df[["pattern", "label"]].copy()
    df["pattern"] = df["pattern"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def _resolve_label(relative_path: str, label_map: pd.DataFrame) -> str | None:
    if not label_map.empty:
        for _, row in label_map.iterrows():
            if re.search(row["pattern"], relative_path):
                return row["label"]
    return _guess_label_from_name(relative_path)


def _extract_time_channels(mat: dict[str, object]) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value).squeeze()
        if arr.ndim != 1 or arr.size < 128:
            continue

        arr = arr.astype(np.float32)
        low = key.lower()
        if low.endswith("_de_time"):
            candidates["DE"] = arr
        elif low.endswith("_fe_time"):
            candidates["FE"] = arr
        elif low.endswith("_ba_time"):
            candidates["BA"] = arr
        elif "time" in low and "DE" not in candidates:
            candidates["DE"] = arr

    if "DE" not in candidates and candidates:
        # Fall back to first available vector if standard channel name is missing.
        first_key = next(iter(candidates.keys()))
        candidates["DE"] = candidates[first_key]

    return candidates


def _to_spectrum(segment: np.ndarray, target_bins: int) -> np.ndarray:
    if segment.ndim != 1:
        raise ValueError("Segment must be 1D")

    window = np.hanning(len(segment)).astype(np.float32)
    spec = np.abs(np.fft.rfft(segment * window)).astype(np.float32)
    spec = spec[1:] if len(spec) > 1 else spec
    spec = np.maximum(spec, 0.0)

    if len(spec) != target_bins:
        src = np.linspace(0.0, 1.0, len(spec), dtype=np.float32)
        dst = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
        spec = np.interp(dst, src, spec).astype(np.float32)

    return spec


def _normalize_tensor(tensor: np.ndarray, log_scale: bool, normalize_per_axis: bool) -> np.ndarray:
    out = tensor.astype(np.float32)
    if log_scale:
        out = np.log1p(np.maximum(out, 0.0))

    if normalize_per_axis:
        for axis_idx in range(out.shape[0]):
            mean = float(out[axis_idx].mean())
            std = float(out[axis_idx].std())
            if std < 1e-6:
                std = 1.0
            out[axis_idx] = (out[axis_idx] - mean) / std
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bootstrap NPZ from Case Western Reserve University (CWRU) bearing .mat files"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing CWRU .mat files")
    parser.add_argument("--output", default="data/bootstrap/cwru_bootstrap.npz", help="Output NPZ path")
    parser.add_argument("--label-map-csv", default="", help="Optional CSV with columns: pattern,label")
    parser.add_argument("--segment-len", type=int, default=4096, help="Segment length in samples")
    parser.add_argument("--segment-step", type=int, default=2048, help="Sliding window step in samples")
    parser.add_argument("--target-bins", type=int, default=2048, help="Output spectrum bins per channel")
    parser.add_argument("--max-segments-per-file", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--log-scale", action="store_true", help="Apply log1p to spectra")
    parser.add_argument("--normalize-per-axis", action="store_true", help="Apply per-axis standardization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = _load_label_map(args.label_map_csv if args.label_map_csv else None)
    mat_files = sorted(input_dir.rglob("*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found under: {input_dir}")

    X_rows: list[np.ndarray] = []
    label_rows: list[str] = []
    run_id_rows: list[str] = []
    operation_rows: list[str] = []
    device_rows: list[str] = []
    source_rows: list[str] = []

    skipped_unlabeled = 0
    skipped_no_channel = 0

    for mat_file in mat_files:
        rel = str(mat_file.relative_to(input_dir)).replace("\\", "/")
        label = _resolve_label(rel, label_map)
        if not label:
            skipped_unlabeled += 1
            continue

        mat = loadmat(mat_file)
        channels = _extract_time_channels(mat)
        if "DE" not in channels:
            skipped_no_channel += 1
            continue

        ch_de = channels["DE"]
        ch_fe = channels.get("FE", ch_de)
        ch_ba = channels.get("BA", ch_de)
        min_len = min(len(ch_de), len(ch_fe), len(ch_ba))
        if min_len < args.segment_len:
            continue

        segment_count = 0
        for start in range(0, min_len - args.segment_len + 1, args.segment_step):
            if args.max_segments_per_file > 0 and segment_count >= args.max_segments_per_file:
                break

            seg_de = ch_de[start : start + args.segment_len]
            seg_fe = ch_fe[start : start + args.segment_len]
            seg_ba = ch_ba[start : start + args.segment_len]

            tensor = np.stack(
                [
                    _to_spectrum(seg_de, args.target_bins),
                    _to_spectrum(seg_fe, args.target_bins),
                    _to_spectrum(seg_ba, args.target_bins),
                ],
                axis=0,
            )
            tensor = _normalize_tensor(
                tensor=tensor,
                log_scale=args.log_scale,
                normalize_per_axis=args.normalize_per_axis,
            )

            run_id = f"cwru-{mat_file.stem}-{segment_count}"
            X_rows.append(tensor)
            label_rows.append(label)
            run_id_rows.append(run_id)
            operation_rows.append("CWRU")
            device_rows.append("CWRU")
            source_rows.append("cwru")
            segment_count += 1

    if not X_rows:
        raise ValueError("No usable CWRU segments were produced. Check label map and file structure.")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    labels = np.asarray(label_rows, dtype=object)
    run_id = np.asarray(run_id_rows, dtype=object)
    operation = np.asarray(operation_rows, dtype=object)
    device_id = np.asarray(device_rows, dtype=object)
    source = np.asarray(source_rows, dtype=object)

    np.savez_compressed(
        output_path,
        X=X,
        labels=labels,
        run_id=run_id,
        operation=operation,
        device_id=device_id,
        source=source,
    )

    class_counts = pd.Series(labels).value_counts().to_dict()
    print(f"Saved bootstrap dataset: {output_path}")
    print(f"Samples: {len(X)} | shape={X.shape[1:]}")
    print(f"Classes: {class_counts}")
    print(f"Skipped unlabeled files: {skipped_unlabeled}")
    print(f"Skipped files without channels: {skipped_no_channel}")


if __name__ == "__main__":
    main()
