from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from vibration_predictor.config import load_config
from vibration_predictor.inference import load_trained_model, predict_probabilities
from vibration_predictor.influx import fetch_frequency_frame
from vibration_predictor.preprocess import attach_labels, build_run_tensors, load_labels_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained CNN on labeled InfluxDB runs")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to trained model checkpoint (.pt). Default: <output_dir>/model.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes, _ = load_trained_model(checkpoint_path, device)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

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

    labels_df = load_labels_csv(cfg.labels_path)
    labeled_meta = attach_labels(metadata, labels_df)
    labeled_mask = labeled_meta["label"].notna().to_numpy()
    if labeled_mask.sum() == 0:
        raise ValueError("No labeled runs found for evaluation")

    X_labeled = X_all[labeled_mask]
    eval_df = labeled_meta.loc[labeled_mask].reset_index(drop=True)

    class_mask = eval_df["label"].isin(classes).to_numpy()
    if class_mask.sum() == 0:
        raise ValueError("None of the labels in data/labels.csv match checkpoint classes")

    X_eval = X_labeled[class_mask]
    eval_df = eval_df.loc[class_mask].reset_index(drop=True)
    y_true = eval_df["label"].map(class_to_idx).to_numpy(dtype=np.int64)

    probs = predict_probabilities(model, X_eval, device=device, batch_size=cfg.training.batch_size)
    y_pred = probs.argmax(axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print("Evaluation classification report:")
    print(report_text)

    pred_df = eval_df[["operation", "run_id", "label"]].copy()
    pred_df["predicted_label"] = [classes[i] for i in y_pred]
    pred_df["confidence"] = probs.max(axis=1)
    pred_df["is_correct"] = pred_df["label"].to_numpy() == pred_df["predicted_label"].to_numpy()
    for idx, class_name in enumerate(classes):
        pred_df[f"prob_{class_name}"] = probs[:, idx]

    pred_path = output_dir / "evaluation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / "evaluation_confusion_matrix.csv")

    metrics = {
        "num_runs_evaluated": int(len(eval_df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classes": classes,
        "classification_report": report_dict,
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved evaluation outputs to: {output_dir}")
    print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()
