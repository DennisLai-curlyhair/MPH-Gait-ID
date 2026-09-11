"""Restricted loading for gait state-dictionary checkpoints."""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

from packaging.version import Version
import torch


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    # Older restricted loaders are affected by CVE-2025-32434.
    if Version(torch.__version__.split("+")[0]) < Version("2.6.0"):
        raise RuntimeError("Checkpoint loading requires PyTorch >=2.6.0; upgrade PyTorch first")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, TypeError) as exc:
        raise ValueError(
            "Unsupported checkpoint. Supply a state-dictionary checkpoint with numeric tensors "
            "under 'model' and plain config metadata. Unrestricted pickle loading is disabled."
        ) from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint must contain a model state dictionary under key 'model'")
    if not checkpoint["model"] or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        or value.layout != torch.strided or not bool(torch.isfinite(value).all())
        for key, value in checkpoint["model"].items()
    ):
        raise ValueError("Model state must contain finite, dense tensors with string keys")
    return checkpoint
