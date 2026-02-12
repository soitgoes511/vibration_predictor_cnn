from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vibration_predictor.config import ExperimentConfig, load_config
from vibration_predictor.influx import fetch_frequency_frame
from vibration_predictor.model import BearingFaultCNN
from vibration_predictor.preprocess import attach_labels, build_run_tensors, load_labels_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(y: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_split = cfg.training.train_split
    val_split = cfg.training.val_split

    remaining = 1.0 - train_split
    val_ratio_within_remaining = val_split / remaining

    try:
        train_idx, temp_idx = train_test_split(
            all_idx,
            train_size=train_split,
            random_state=cfg.training.random_seed,
            stratify=y,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_within_remaining,
            random_state=cfg.training.random_seed,
            stratify=y[temp_idx],
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            all_idx,
            train_size=train_split,
            random_state=cfg.training.random_seed,
            stratify=None,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_within_remaining,
            random_state=cfg.training.random_seed,
            stratify=None,
        )

    return train_idx, val_idx, test_idx


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            losses.append(float(loss.item()))
            batch_pred = torch.argmax(logits, dim=1)
            preds.extend(batch_pred.cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())

    if not targets:
        return {"loss": float("nan"), "accuracy": float("nan")}, np.array([]), np.array([])

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, preds)),
    }
    return metrics, np.asarray(preds), np.asarray(targets)


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    X_train = torch.from_numpy(X[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()
    X_test = torch.from_numpy(X[test_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    y_val = torch.from_numpy(y[val_idx]).long()
    y_test = torch.from_numpy(y[test_idx]).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bearing fault CNN from InfluxDB FFT data")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.training.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Fetching frequency data from InfluxDB...")
    freq_df = fetch_frequency_frame(cfg.influx)
    if freq_df.empty:
        raise ValueError("No rows returned from InfluxDB query")
    print(f"Fetched {len(freq_df)} frequency rows")

    X_all, metadata = build_run_tensors(
        freq_df=freq_df,
        target_bins=cfg.dataset.target_bins,
        min_bins_per_run=cfg.dataset.min_bins_per_run,
        log_scale=cfg.dataset.log_scale,
        normalize_per_axis=cfg.dataset.normalize_per_axis,
    )
    print(f"Prepared {len(metadata)} run tensors with shape {X_all.shape[1:]}")

    labels_df = load_labels_csv(cfg.labels_path)
    labeled_meta = attach_labels(metadata, labels_df)
    labeled_mask = labeled_meta["label"].notna().to_numpy()
    if labeled_mask.sum() < 10:
        raise ValueError("Too few labeled runs. Add more labels to data/labels.csv")

    X = X_all[labeled_mask]
    dataset_df = labeled_meta.loc[labeled_mask].reset_index(drop=True)
    y_text = dataset_df["label"].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    num_classes = len(encoder.classes_)
    if num_classes < 2:
        raise ValueError("Need at least two classes for classification")

    train_idx, val_idx, test_idx = split_indices(y, cfg)
    print(
        f"Runs split -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, classes={num_classes}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        X=X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=cfg.training.batch_size,
    )

    model = BearingFaultCNN(
        num_classes=num_classes,
        conv_channels=cfg.model.conv_channels,
        kernel_sizes=cfg.model.kernel_sizes,
        dropout=cfg.model.dropout,
    ).to(device)

    class_counts = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.clip(class_counts, a_min=1.0, a_max=None)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        train_preds: list[int] = []
        train_targets: list[int] = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            train_targets.extend(batch_y.detach().cpu().numpy().tolist())

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(accuracy_score(train_targets, train_preds)) if train_targets else float("nan")
        val_metrics, _, _ = evaluate_loader(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"train_acc={train_acc:.4f} val_loss={val_metrics['loss']:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = float(val_metrics["loss"])
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics, test_preds, test_targets = evaluate_loader(model, test_loader, criterion, device)
    macro_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))
    report_text = classification_report(
        test_targets,
        test_preds,
        labels=list(range(num_classes)),
        target_names=encoder.classes_,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        test_targets,
        test_preds,
        labels=list(range(num_classes)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))

    print("Test classification report:")
    print(report_text)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": encoder.classes_.tolist(),
        "model_hparams": {
            "conv_channels": list(cfg.model.conv_channels),
            "kernel_sizes": list(cfg.model.kernel_sizes),
            "dropout": cfg.model.dropout,
        },
        "dataset_hparams": {
            "target_bins": cfg.dataset.target_bins,
            "min_bins_per_run": cfg.dataset.min_bins_per_run,
            "log_scale": cfg.dataset.log_scale,
            "normalize_per_axis": cfg.dataset.normalize_per_axis,
        },
        "training_hparams": {
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
        },
        "config_snapshot": asdict(cfg),
    }
    torch.save(checkpoint, output_dir / "model.pt")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "history.csv", index=False)

    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    split = np.full(len(dataset_df), "unused", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    dataset_df["split"] = split
    dataset_df["encoded_label"] = y
    dataset_df.to_csv(output_dir / "dataset_index.csv", index=False)

    test_df = dataset_df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["predicted_label"] = encoder.inverse_transform(test_preds)
    test_df["is_correct"] = test_df["label"].to_numpy() == test_df["predicted_label"].to_numpy()
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics_payload = {
        "num_runs_unlabeled_ignored": int((~labeled_mask).sum()),
        "num_runs_labeled_total": int(len(dataset_df)),
        "num_classes": int(num_classes),
        "classes": encoder.classes_.tolist(),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(macro_f1),
        "classification_report": report_dict,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
