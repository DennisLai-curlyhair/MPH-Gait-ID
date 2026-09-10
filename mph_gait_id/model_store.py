from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import SYSTEM_ROOT


DEFAULT_BUNDLE_ROOT = SYSTEM_ROOT / "model_bundles"
SUPPORTED_ARCHITECTURES = {
    "temporal_pointnet_baseline",
    "pc_v1",
    "mph_gait",
    "lidargaitpp_official",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class ModelBundle:
    bundle_id: str
    display_name: str
    method_key: str
    description: str
    architecture: str
    input_type: str
    mode: str
    channels: int
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    checkpoint: Path
    checkpoint_sha256: str
    manifest_path: Path

    @property
    def fold(self) -> int:
        return int(self.training.get("fold", -1))

    @property
    def key(self) -> str:
        return self.bundle_id

    @property
    def bundle_dir(self) -> Path:
        return self.manifest_path.parent

    def public_record(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "display_name": self.display_name,
            "method_key": self.method_key,
            "description": self.description,
            "architecture": self.architecture,
            "input_type": self.input_type,
            "mode": self.mode,
            "channels": self.channels,
            "fold": self.fold,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "clip_len": int(self.data.get("clip_len", 15)),
            "drop_first_frames": int(self.data.get("drop_first_frames", 30)),
            "num_points": self.data.get("num_points"),
        }


def descriptor_dimension(bundle: ModelBundle) -> int:
    """Read the flattened inference dimension, including official part heads."""
    if bundle.architecture == "lidargaitpp_official":
        head = bundle.model.get("SeparateFCs", {})
        channels = int(head.get("out_channels", bundle.model.get("embedding_dim", 256)))
        parts = int(head.get("parts_num", bundle.model.get("parts_num", 31)))
        return channels * parts
    return int(bundle.model.get("embedding_dim", 256))


class ModelStore:
    """Discovers, verifies, and imports self-contained model bundles."""

    def __init__(self, root: str | Path = DEFAULT_BUNDLE_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._bundles: dict[str, ModelBundle] = {}
        self.refresh()

    def refresh(self) -> dict[str, ModelBundle]:
        bundles: dict[str, ModelBundle] = {}
        for manifest_path in sorted(self.root.glob("*/bundle.yaml")):
            bundle = self._load_manifest(manifest_path)
            if bundle.bundle_id in bundles:
                raise ValueError(f"Duplicate model bundle id: {bundle.bundle_id}")
            bundles[bundle.bundle_id] = bundle
        self._bundles = bundles
        return dict(self._bundles)

    def bundles(self) -> dict[str, ModelBundle]:
        return dict(self._bundles)

    def get(self, bundle_id: str) -> ModelBundle:
        if bundle_id not in self._bundles:
            available = ", ".join(self._bundles) or "none"
            raise KeyError(f"Unknown model bundle '{bundle_id}'. Available: {available}")
        return self._bundles[bundle_id]

    def find_legacy(self, method_key: str, fold: int = 0) -> ModelBundle:
        matches = [
            bundle
            for bundle in self._bundles.values()
            if bundle.method_key == method_key and bundle.fold == int(fold)
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one bundle for {method_key} fold {fold}, found {len(matches)}")
        return matches[0]

    def verify(self, bundle_id: str, check_hash: bool = True) -> dict[str, Any]:
        bundle = self.get(bundle_id)
        errors: list[str] = []
        actual_hash = None
        if not bundle.checkpoint.is_file():
            errors.append("checkpoint is missing")
        elif check_hash:
            actual_hash = sha256_file(bundle.checkpoint)
            if actual_hash != bundle.checkpoint_sha256:
                errors.append("checkpoint SHA256 does not match bundle.yaml")
        return {
            **bundle.public_record(),
            "actual_sha256": actual_hash,
            "valid": not errors,
            "errors": errors,
        }

    def import_checkpoint(
        self,
        checkpoint_path: str | Path,
        display_name: str,
        template_bundle_id: str | None = None,
    ) -> ModelBundle:
        source = Path(checkpoint_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {source}")
        checkpoint = _torch_load(source)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("Checkpoint must contain a model state dictionary under key 'model'")

        template = self.get(template_bundle_id) if template_bundle_id else None
        raw_config = checkpoint.get("config", {})
        if not isinstance(raw_config, dict):
            raw_config = {}
        checkpoint_model = raw_config.get("model", {})
        checkpoint_data = raw_config.get("data", {})
        if not isinstance(checkpoint_model, dict):
            checkpoint_model = {}
        if not isinstance(checkpoint_data, dict):
            checkpoint_data = {}
        model = dict(template.model if template else {})
        model.update(checkpoint_model)
        data = dict(template.data if template else {})
        data.update(checkpoint_data)
        train = dict(raw_config.get("train") or {})
        configuration_source = (
            "checkpoint_with_template_defaults"
            if checkpoint_model or checkpoint_data
            else "selected_template"
        )
        if template is not None:
            # A training config's model.name is the experiment/model class name
            # (for example pc_v1_temporal_pointnet), not this application's
            # runtime adapter identifier. The selected bundle is the explicit
            # compatibility contract for an imported checkpoint.
            architecture = template.architecture
        else:
            configured_name = str(model.get("name", ""))
            architecture = {
                "pc_v1_temporal_pointnet": "pc_v1",
                "pc_v1_hierarchical_multiscale_regular_transformer_v1": "mph_gait",
                "lidargait_plus_plus": "lidargaitpp_official",
                "LidarGaitPlusPlus": "lidargaitpp_official",
            }.get(configured_name, configured_name)
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported model architecture '{architecture}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}"
            )
        if template is not None and architecture != template.architecture:
            raise ValueError(
                f"Checkpoint architecture '{architecture}' does not match selected template "
                f"'{template.architecture}'"
            )

        input_type = "pointcloud"
        mode = "raw_pointcloud"
        channels = 4 if architecture == "lidargaitpp_official" else int(
            model.get("in_channels", 3)
        )
        method_key = template.method_key if template else "pointcloud_custom"
        clip_len = int(data.get("clip_len", 0))
        if clip_len <= 0:
            raise ValueError(
                "Checkpoint/Bundle 缺少有效的 data.clip_len；此參數無法由權重張量自動推測。"
            )
        if int(data.get("num_points", 0)) <= 0:
            raise ValueError(
                "Point-cloud checkpoint/Bundle 缺少有效的 data.num_points；"
                "請選擇與訓練設定相符的模板。"
            )
        source_hash = sha256_file(source)
        slug = re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_") or "imported_model"
        bundle_id = f"user_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source_hash[:8]}"
        target_dir = self.root / bundle_id
        target_dir.mkdir(parents=False, exist_ok=False)
        target_checkpoint = target_dir / "checkpoint.pt"
        try:
            shutil.copy2(source, target_checkpoint)
            if sha256_file(target_checkpoint) != source_hash:
                raise RuntimeError("Copied checkpoint failed SHA256 verification")
            manifest = {
                "schema_version": 1,
                "bundle_id": bundle_id,
                "display_name": display_name,
                "method_key": method_key,
                "description": "User-imported checkpoint stored inside the application model store.",
                "architecture": architecture,
                "input": {
                    "type": input_type,
                    "mode": mode,
                    "channels": channels,
                },
                "data": {
                    key: data[key]
                    for key in (
                        "num_points",
                        "clip_len",
                        "drop_first_frames",
                        "batch_size",
                        "coordinate_adapter",
                    )
                    if key in data
                },
                "model": model,
                "training": {
                    "fold": int(raw_config.get("protocol", {}).get("fold_index", -1)),
                    "epoch": int(checkpoint.get("epoch", -1)),
                    "best_val_mAP": float(checkpoint.get("best_metric", float("nan"))),
                    "amp": bool(train.get("amp", False)),
                    "imported_from": str(source),
                    "configuration_source": configuration_source,
                },
                "checkpoint": {"file": "checkpoint.pt", "sha256": source_hash},
            }
            (target_dir / "bundle.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        self.refresh()
        return self.get(bundle_id)

    def _load_manifest(self, manifest_path: Path) -> ModelBundle:
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Bundle manifest must be a mapping: {manifest_path}")
        input_config = raw.get("input", {})
        checkpoint_config = raw.get("checkpoint", {})
        checkpoint = (manifest_path.parent / str(checkpoint_config.get("file", "checkpoint.pt"))).resolve()
        try:
            checkpoint.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"Checkpoint must stay inside its bundle directory: {manifest_path}") from exc
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Bundle checkpoint is missing: {checkpoint}")
        architecture = str(raw.get("architecture", ""))
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture in {manifest_path}: {architecture}")
        input_type = str(input_config.get("type", "pointcloud"))
        mode = str(input_config.get("mode", "raw_pointcloud"))
        if input_type != "pointcloud" or mode != "raw_pointcloud":
            raise ValueError(
                f"This application accepts raw point-cloud model bundles only: {manifest_path}"
            )
        expected_hash = str(checkpoint_config.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Invalid checkpoint SHA256 in {manifest_path}")
        return ModelBundle(
            bundle_id=str(raw["bundle_id"]),
            display_name=str(raw.get("display_name", raw["bundle_id"])),
            method_key=str(raw.get("method_key", raw["bundle_id"])),
            description=str(raw.get("description", "")),
            architecture=architecture,
            input_type=input_type,
            mode=mode,
            channels=int(input_config.get("channels", 3)),
            data=dict(raw.get("data", {})),
            model=dict(raw.get("model", {})),
            training=dict(raw.get("training", {})),
            checkpoint=checkpoint,
            checkpoint_sha256=expected_hash,
            manifest_path=manifest_path.resolve(),
        )
