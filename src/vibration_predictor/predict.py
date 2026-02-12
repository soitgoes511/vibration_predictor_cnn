from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from vibration_predictor.config import load_config
from vibration_predictor.inference import load_trained_model, predict_probabilities
from vibration_predictor.influx import fetch_frequency_frame
from vibration_predictor.preprocess import build_run_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bearing fault class for InfluxDB runs")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to trained model checkpoint (.pt). Default: <output_dir>/model.pt",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Specific run_id to score (repeat for multiple IDs)",
    )
    parser.add_argument(
        "--operation",
        default="",
        help="Override operation filter from config for this prediction run",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default: <output_dir>/predictions.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of rows to print in console summary",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.operation:
        cfg = replace(cfg, influx=replace(cfg.influx, operation=args.operation))

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "predictions.csv"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, _ = load_trained_model(checkpoint_path, device)

    print("Fetching frequency data from InfluxDB...")
    freq_df = fetch_frequency_frame(cfg.influx)
    if freq_df.empty:
        raise ValueError("No rows returned from InfluxDB query")

    X_all, metadata = build_run_tensors(
        freq_df=freq_df,
        target_bins=cfg.dataset.target_bins,
        min_bins_per_run=cfg.dataset.min_bins_per_run,
        log_scale=cfg.dataset.log_scale,
        normalize_per_axis=cfg.dataset.normalize_per_axis,
    )

    if args.run_id:
        requested = {str(run_id) for run_id in args.run_id}
        mask = metadata["run_id"].isin(requested).to_numpy()
        if mask.sum() == 0:
            raise ValueError("No requested run_id values found in queried Influx data")
        X = X_all[mask]
        meta = metadata.loc[mask].reset_index(drop=True)
    else:
        X = X_all
        meta = metadata.reset_index(drop=True)

    probs = predict_probabilities(model, X, device=device, batch_size=cfg.training.batch_size)
    pred_idx = probs.argmax(axis=1)

    pred_df = meta[["operation", "run_id", "source_bins", "min_frequency_hz", "max_frequency_hz"]].copy()
    pred_df["predicted_label"] = [classes[i] for i in pred_idx]
    pred_df["confidence"] = probs.max(axis=1)
    for idx, class_name in enumerate(classes):
        pred_df[f"prob_{class_name}"] = probs[:, idx]

    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

    preview = pred_df.sort_values("confidence", ascending=False).head(max(args.limit, 1))
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(preview)


if __name__ == "__main__":
    main()
