from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model_store import ModelBundle, sha256_file
from .checkpoint_io import load_checkpoint as _torch_load
from .models.lidargaitpp import build_model as build_lidargaitpp_model
from .models.mph_gait import build_model as build_mph_model
from .models.pc_v1 import build_model as build_pc_v1_model
from .models.pointcloud import build_model as build_pointcloud_model


def resolve_device(device_name: str) -> torch.device:
    name = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if name == "auto":
        name = "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class ModelAdapter:
    """Loads one local model bundle and exposes normalized embeddings."""

    def __init__(self, bundle: ModelBundle, device: str = "auto", amp: bool | None = None) -> None:
        actual_hash = sha256_file(bundle.checkpoint)
        if actual_hash != bundle.checkpoint_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: {bundle.checkpoint}")
        self.bundle = bundle
        self.device = resolve_device(device)
        checkpoint = _torch_load(bundle.checkpoint, map_location=self.device)
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        checkpoint_model = checkpoint_config.get("model", {}) if isinstance(checkpoint_config, dict) else {}
        if not isinstance(checkpoint_model, dict):
            raise ValueError("Checkpoint model config must be a mapping")
        if any(value != bundle.model.get(key) for key, value in checkpoint_model.items()):
            raise ValueError("Checkpoint model config differs from bundle.yaml; re-import a matching bundle")
        model_config = dict(bundle.model)
        if bundle.input_type != "pointcloud":
            raise ValueError(
                "This application only supports point-cloud model bundles, "
                f"got: {bundle.input_type}"
            )
        if bundle.architecture == "pc_v1":
            self.model = build_pc_v1_model(model_config).to(self.device)
        elif bundle.architecture == "mph_gait":
            self.model = build_mph_model(model_config).to(self.device)
        elif bundle.architecture == "lidargaitpp_official":
            self.model = build_lidargaitpp_model(model_config).to(self.device)
        elif bundle.architecture == "temporal_pointnet_baseline":
            model_config["in_channels"] = int(model_config.get("in_channels", bundle.channels))
            self.model = build_pointcloud_model(model_config).to(self.device)
        else:
            raise ValueError(f"Unsupported bundle architecture: {bundle.architecture}")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.training_amp = bool(bundle.training.get("amp", False))
        requested_amp = False if amp is None else bool(amp)
        self.amp_enabled = requested_amp and self.device.type == "cuda"
        self.fp32_fallback_count = 0
        self.epoch = int(checkpoint.get("epoch", bundle.training.get("epoch", -1)))
        self.best_metric = float(
            checkpoint.get("best_metric", bundle.training.get("best_val_mAP", float("nan")))
        )

    def make_loader(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> DataLoader:
        configured_batch = int(self.bundle.data.get("batch_size", 16))
        return DataLoader(
            dataset,
            batch_size=int(batch_size or configured_batch),
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    @torch.no_grad()
    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> torch.Tensor:
        loader = self.make_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        embeddings: list[torch.Tensor] = []
        for batch_index, batch in enumerate(loader):
            cpu_inputs = batch["points"]
            if not bool(torch.isfinite(cpu_inputs).all().item()):
                raise ValueError(
                    f"Input batch {batch_index} contains NaN or Inf values before model inference"
                )
            inputs = cpu_inputs.to(self.device, non_blocking=True)
            raw_embedding = self._forward_embedding(inputs, use_amp=self.amp_enabled)
            if not bool(torch.isfinite(raw_embedding).all().item()) and self.amp_enabled:
                raw_embedding = self._forward_embedding(inputs, use_amp=False)
                self.fp32_fallback_count += 1
            if not bool(torch.isfinite(raw_embedding).all().item()):
                raise RuntimeError(
                    f"The model produced NaN or Inf embeddings in batch {batch_index}, "
                    "including after FP32 inference. Verify that the selected folder contains "
                    f"{self.bundle.mode} data produced by the expected preprocessing pipeline."
                )
            if bool((raw_embedding.norm(dim=1) <= 1e-12).any()):
                raise RuntimeError(f"The model produced a zero-length descriptor in batch {batch_index}")
            embedding = F.normalize(raw_embedding.cpu(), dim=1)
            if not bool(torch.isfinite(embedding).all().item()):
                raise RuntimeError(
                    f"Embedding normalization produced NaN or Inf in batch {batch_index}"
                )
            embeddings.append(embedding)
        if not embeddings:
            return torch.empty(0, 1)
        return torch.cat(embeddings, dim=0)

    def _forward_embedding(self, inputs: torch.Tensor, use_amp: bool) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)
        with autocast_context(self.device, use_amp):
            output = self.model(inputs)
        return output["embedding"].detach().float()

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float()
        if values.ndim != 4 or values.shape[-1] < 3:
            raise ValueError(
                f"Expected point cloud [B,T,N,C>=3], got {tuple(values.shape)}"
            )
        values = values[..., :3]
        if min(values.shape[:3]) == 0 or values.shape[2] < 2:
            raise ValueError("Point-cloud batches require nonempty windows and at least two points")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Point-cloud input contains NaN or Inf")
        spread = values.amax(dim=2) - values.amin(dim=2)
        if bool((spread.abs().amax(dim=-1) == 0).any()):
            raise ValueError("Point-cloud input contains an empty or degenerate frame")
        adapter = str(self.bundle.data.get("coordinate_adapter", "none"))
        if adapter == "kinect_xyz_mm_to_forward_lateral_height_m_v1":
            x, y, z = values.unbind(dim=-1)
            values = torch.stack((z, x, -y), dim=-1) * 0.001
        elif adapter not in {"none", "identity", "axis_aligned_m"}:
            raise ValueError(f"Unsupported coordinate adapter: {adapter}")

        if self.bundle.architecture == "lidargaitpp_official":
            heights = values[..., 2:3] - values[..., 2:3].amin(
                dim=2, keepdim=True
            )
            centered = values - values.mean(dim=2, keepdim=True)
            scale = centered.norm(dim=-1).amax(dim=2, keepdim=True).unsqueeze(-1)
            normalized = centered / scale.clamp_min(1e-6)
            values = torch.cat((normalized, heights), dim=-1)
        return values

    @torch.no_grad()
    def encode_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode one or more already-windowed live sequences."""

        values = torch.as_tensor(inputs)
        if values.ndim == 3:
            values = values.unsqueeze(0)
        if values.ndim != 4:
            raise ValueError("Point-cloud live input must be [T,N,C] or [B,T,N,C]")
        values = values.to(self.device, non_blocking=True)
        raw = self._forward_embedding(values, use_amp=self.amp_enabled)
        if not bool(torch.isfinite(raw).all().item()) and self.amp_enabled:
            raw = self._forward_embedding(values, use_amp=False)
            self.fp32_fallback_count += 1
        if not bool(torch.isfinite(raw).all().item()):
            raise RuntimeError("The live model produced NaN or Inf embeddings")
        if bool((raw.norm(dim=1) <= 1e-12).any()):
            raise RuntimeError("The live model produced a zero-length descriptor")
        return F.normalize(raw.float(), dim=1).cpu()

    def manifest(self) -> dict[str, Any]:
        return {
            **self.bundle.public_record(),
            "checkpoint_epoch": self.epoch,
            "checkpoint_best_val_mAP": self.best_metric,
            "device": str(self.device),
            "amp": self.amp_enabled,
            "training_amp": self.training_amp,
            "fp32_fallback_count": self.fp32_fallback_count,
        }
