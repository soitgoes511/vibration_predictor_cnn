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

from vibration_predictor.bootstrap_data import load_bootstrap_npz
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


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _resample_bins(X: np.ndarray, target_bins: int) -> np.ndarray:
    if X.shape[2] == target_bins:
        return X.astype(np.float32)

    src = np.linspace(0.0, 1.0, X.shape[2], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    out = np.empty((X.shape[0], X.shape[1], target_bins), dtype=np.float32)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            out[i, ch] = np.interp(dst, src, X[i, ch]).astype(np.float32)
    return out


def _load_influx_labeled_dataset(cfg: ExperimentConfig) -> tuple[np.ndarray, pd.DataFrame, int]:
    if not cfg.influx.token:
        if cfg.bootstrap.enabled:
            print("Influx token not provided; skipping Influx data load because bootstrap is enabled")
            empty = np.empty((0, 3, cfg.dataset.target_bins), dtype=np.float32)
            return empty, pd.DataFrame(columns=["operation", "run_id", "label", "source"]), 0
        raise ValueError("Influx token is empty. Set INFLUXDB_TOKEN or influx.token in config")

    print("Fetching frequency data from InfluxDB...")
    freq_df = fetch_frequency_frame(cfg.influx)
    if freq_df.empty:
        print("No Influx frequency rows returned")
        empty = np.empty((0, 3, cfg.dataset.target_bins), dtype=np.float32)
        return empty, pd.DataFrame(columns=["operation", "run_id", "label", "source"]), 0

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

    if labeled_mask.sum() == 0:
        empty = np.empty((0, 3, cfg.dataset.target_bins), dtype=np.float32)
        return empty, pd.DataFrame(columns=["operation", "run_id", "label", "source"]), int((~labeled_mask).sum())

    X = X_all[labeled_mask]
    dataset_df = labeled_meta.loc[labeled_mask].reset_index(drop=True)
    dataset_df["source"] = "influx"
    return X, dataset_df, int((~labeled_mask).sum())


def _load_bootstrap_dataset(cfg: ExperimentConfig) -> tuple[np.ndarray, pd.DataFrame]:
    if not cfg.bootstrap.enabled:
        empty = np.empty((0, 3, cfg.dataset.target_bins), dtype=np.float32)
        return empty, pd.DataFrame(columns=["operation", "run_id", "label", "source", "device_id"])

    X_boot, labels_boot, meta_boot = load_bootstrap_npz(
        path=cfg.bootstrap.npz_path,
        max_samples_per_class=cfg.bootstrap.max_samples_per_class,
        random_seed=cfg.training.random_seed,
    )
    X_boot = _resample_bins(X_boot, cfg.dataset.target_bins)

    if meta_boot.empty:
        meta_boot = pd.DataFrame(index=np.arange(len(X_boot)))
    meta_boot = meta_boot.copy()
    meta_boot["label"] = labels_boot
    meta_boot["source"] = meta_boot.get("source", "bootstrap")
    meta_boot["operation"] = meta_boot.get("operation", "BOOTSTRAP")
    meta_boot["run_id"] = meta_boot.get("run_id", [f"bootstrap_{i}" for i in range(len(X_boot))])
    meta_boot["device_id"] = meta_boot.get("device_id", "BOOTSTRAP")

    print(f"Loaded bootstrap samples: {len(X_boot)} from {cfg.bootstrap.npz_path}")
    return X_boot, meta_boot.reset_index(drop=True)


def _prepare_datasets(
    cfg: ExperimentConfig,
    X_influx: np.ndarray,
    influx_df: pd.DataFrame,
    X_boot: np.ndarray,
    boot_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    str,
]:
    influx_labels = influx_df["label"].astype(str).to_numpy() if not influx_df.empty else np.asarray([], dtype=str)
    boot_labels = boot_df["label"].astype(str).to_numpy() if not boot_df.empty else np.asarray([], dtype=str)

    label_text_all = np.concatenate([influx_labels, boot_labels])
    if label_text_all.size == 0:
        raise ValueError("No labeled data available. Add labels.csv and/or enable bootstrap NPZ dataset.")

    encoder = LabelEncoder()
    y_encoded_all = encoder.fit_transform(label_text_all)
    if len(encoder.classes_) < 2:
        raise ValueError("Need at least two classes for classification")

    n_influx = len(influx_labels)
    y_influx = y_encoded_all[:n_influx]
    y_boot = y_encoded_all[n_influx:]

    split_mode = "combined"
    use_train_only = cfg.bootstrap.enabled and cfg.bootstrap.use_for_training_only and len(X_boot) > 0 and len(X_influx) > 0
    if use_train_only and len(X_influx) < 10:
        # Too little real data to create stable val/test splits.
        use_train_only = False

    if use_train_only:
        split_mode = "influx_plus_bootstrap_train_only"
        train_idx, val_idx, test_idx = split_indices(y_influx, cfg)

        X_train = np.concatenate([X_influx[train_idx], X_boot], axis=0)
        y_train = np.concatenate([y_influx[train_idx], y_boot], axis=0)
        X_val = X_influx[val_idx]
        y_val = y_influx[val_idx]
        X_test = X_influx[test_idx]
        y_test = y_influx[test_idx]

        influx_export = influx_df.copy()
        boot_export = boot_df.copy()
        influx_export["split"] = "unused"
        influx_export.loc[train_idx, "split"] = "train"
        influx_export.loc[val_idx, "split"] = "val"
        influx_export.loc[test_idx, "split"] = "test"
        boot_export["split"] = "train_bootstrap"
        dataset_export = pd.concat([influx_export, boot_export], ignore_index=True)

        test_rows = influx_export.iloc[test_idx].copy().reset_index(drop=True)
    else:
        X_combined = np.concatenate([X_influx, X_boot], axis=0) if len(X_boot) > 0 else X_influx.copy()
        y_combined = np.concatenate([y_influx, y_boot], axis=0) if len(y_boot) > 0 else y_influx.copy()

        if len(X_combined) < 10:
            raise ValueError("Too few labeled samples for training. Need at least 10.")

        train_idx, val_idx, test_idx = split_indices(y_combined, cfg)
        X_train = X_combined[train_idx]
        y_train = y_combined[train_idx]
        X_val = X_combined[val_idx]
        y_val = y_combined[val_idx]
        X_test = X_combined[test_idx]
        y_test = y_combined[test_idx]

        dataset_export = pd.concat([influx_df, boot_df], ignore_index=True)
        if dataset_export.empty:
            dataset_export = pd.DataFrame(index=np.arange(len(X_combined)))
        dataset_export["split"] = "unused"
        dataset_export.loc[train_idx, "split"] = "train"
        dataset_export.loc[val_idx, "split"] = "val"
        dataset_export.loc[test_idx, "split"] = "test"

        test_rows = dataset_export.iloc[test_idx].copy().reset_index(drop=True)

    label_to_idx = {label: idx for idx, label in enumerate(encoder.classes_)}
    dataset_export["encoded_label"] = dataset_export["label"].astype(str).map(label_to_idx).fillna(-1).astype(int)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset_export.reset_index(drop=True),
        test_rows,
        encoder.classes_,
        split_mode,
    )


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

    X_influx, influx_df, num_influx_unlabeled = _load_influx_labeled_dataset(cfg)
    X_boot, boot_df = _load_bootstrap_dataset(cfg)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dataset_export,
        test_rows,
        classes,
        split_mode,
    ) = _prepare_datasets(cfg, X_influx, influx_df, X_boot, boot_df)

    print(
        f"Split mode: {split_mode} | train={len(X_train)} val={len(X_val)} test={len(X_test)} classes={len(classes)}"
    )

    train_loader = _build_loader(X_train, y_train, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = _build_loader(X_test, y_test, batch_size=cfg.training.batch_size, shuffle=False)

    model = BearingFaultCNN(
        num_classes=len(classes),
        conv_channels=cfg.model.conv_channels,
        kernel_sizes=cfg.model.kernel_sizes,
        dropout=cfg.model.dropout,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
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
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        test_targets,
        test_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(classes))))

    print("Test classification report:")
    print(report_text)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes.tolist(),
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

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    dataset_export.to_csv(output_dir / "dataset_index.csv", index=False)

    test_df = test_rows.copy()
    test_df["predicted_label"] = np.asarray(classes)[test_preds]
    test_df["is_correct"] = test_df["label"].astype(str).to_numpy() == test_df["predicted_label"].to_numpy()
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics_payload = {
        "split_mode": split_mode,
        "num_influx_unlabeled_ignored": int(num_influx_unlabeled),
        "num_influx_labeled_used": int(len(influx_df)),
        "num_bootstrap_labeled_used": int(len(boot_df)),
        "num_samples_total": int(len(dataset_export)),
        "num_classes": int(len(classes)),
        "classes": classes.tolist(),
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
