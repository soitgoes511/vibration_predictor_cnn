from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vibration_predictor.model import BearingFaultCNN


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[BearingFaultCNN, list[str], dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in checkpoint or "classes" not in checkpoint:
        raise ValueError("Invalid checkpoint format: missing model_state_dict/classes")

    classes = [str(c) for c in checkpoint["classes"]]
    hparams = checkpoint.get("model_hparams", {})
    conv_channels = tuple(int(v) for v in hparams.get("conv_channels", [32, 64, 128]))
    kernel_sizes = tuple(int(v) for v in hparams.get("kernel_sizes", [7, 5, 3]))
    dropout = float(hparams.get("dropout", 0.3))

    model = BearingFaultCNN(
        num_classes=len(classes),
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes, checkpoint


def predict_probabilities(
    model: BearingFaultCNN,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    if inputs.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    tensor = torch.from_numpy(inputs).float()
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(batch_probs)

    return np.concatenate(probs, axis=0) if probs else np.empty((0, 0), dtype=np.float32)
