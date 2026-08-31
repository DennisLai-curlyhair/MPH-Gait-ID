from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataio import build_window_dataset
from .identity import parse_sequence_metadata
from .model_adapter import ModelAdapter
from .model_store import ModelBundle, ModelStore
from .source_fingerprint import compute_source_fingerprint


@dataclass
class EmbeddingBatch:
    embeddings: np.ndarray
    windows: list[dict[str, Any]]
    source: Path
    source_metadata: dict[str, Any]
    model: dict[str, Any]


def _sequence_metadata(path: Path) -> dict[str, Any]:
    return parse_sequence_metadata(path)


class SystemModelRuntime:
    """Loads and runs one model bundle fully contained in this application folder."""

    def __init__(
        self,
        bundle_id: str | None = None,
        device: str = "auto",
        batch_size: int | None = None,
        num_workers: int = 0,
        model_store: ModelStore | None = None,
        method_key: str | None = None,
        fold: int = 0,
        runtime_clip_len: int | None = None,
        runtime_drop_first_frames: int | None = None,
        preprocessing_profile_id: str = "person_foreground_pointcloud_v1",
    ) -> None:
        self.model_store = model_store or ModelStore()
        if bundle_id is None:
            if method_key is None:
                raise ValueError("bundle_id is required")
            bundle_id = self.model_store.find_legacy(method_key, fold).bundle_id
        self.bundle: ModelBundle = self.model_store.get(bundle_id)
        self.method = self.bundle
        self.fold = self.bundle.fold
        self.runtime_clip_len = (
            int(runtime_clip_len)
            if runtime_clip_len is not None
            else int(self.bundle.data.get("clip_len", 15))
        )
        if self.runtime_clip_len <= 0:
            raise ValueError("runtime_clip_len must be positive")
        self.runtime_drop_first_frames = (
            int(runtime_drop_first_frames)
            if runtime_drop_first_frames is not None
            else int(self.bundle.data.get("drop_first_frames", 30))
        )
        if self.runtime_drop_first_frames < 0:
            raise ValueError("runtime_drop_first_frames must not be negative")
        self.preprocessing_profile_id = str(preprocessing_profile_id).strip()
        if not self.preprocessing_profile_id:
            raise ValueError("preprocessing_profile_id must not be empty")
        self.adapter = ModelAdapter(self.bundle, device=device)
        self.batch_size = batch_size
        self.num_workers = max(0, int(num_workers))
        self.model_record = self._build_model_record()

    def _configured_embedding_dim(self) -> int:
        embedding_dim = int(self.bundle.model.get("embedding_dim", 256))
        if self.bundle.architecture == "lidargaitpp_official":
            embedding_dim *= int(self.bundle.model.get("parts_num", 1))
        return embedding_dim

    def _build_model_record(self, embedding_dim: int | None = None) -> dict[str, Any]:
        manifest = self.adapter.manifest()
        embedding_dim = int(embedding_dim or self._configured_embedding_dim())
        try:
            checkpoint_path = str(
                self.bundle.checkpoint.relative_to(self.model_store.root.parent)
            )
        except ValueError:
            checkpoint_path = str(self.bundle.checkpoint)
        compatibility = {
            "bundle_id": self.bundle.bundle_id,
            "method_key": self.bundle.method_key,
            "architecture": self.bundle.architecture,
            "input_type": self.bundle.input_type,
            "input_mode": self.bundle.mode,
            "checkpoint_sha256": self.bundle.checkpoint_sha256,
            # Sequence length changes the deployment embedding distribution even
            # when the architecture accepts a variable T. Keep T=15 and T=30
            # Gallery records in different compatibility spaces.
            "clip_len": self.runtime_clip_len,
            "drop_first_frames": self.runtime_drop_first_frames,
            "embedding_dim": embedding_dim,
            "num_points": manifest.get("num_points"),
            "point_normalization": self.bundle.model.get("point_normalization"),
            "preprocessing_profile_id": self.preprocessing_profile_id,
        }
        digest = hashlib.sha256(
            json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "model_key": f"{self.bundle.bundle_id}_{digest[:16]}",
            "bundle_id": self.bundle.bundle_id,
            "method_key": self.bundle.method_key,
            "display_name": self.bundle.display_name,
            "architecture": self.bundle.architecture,
            "input_type": self.bundle.input_type,
            "input_mode": self.bundle.mode,
            "fold": self.bundle.fold,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": self.bundle.checkpoint_sha256,
            "embedding_dim": embedding_dim,
            "compatibility": compatibility,
            "research_fold_checkpoint": self.bundle.fold >= 0,
        }

    def extract_folder(
        self,
        source: str | Path,
        window_size: int | None = None,
        stride: int = 5,
        drop_first_frames: int | None = None,
        source_fingerprint: str | None = None,
    ) -> EmbeddingBatch:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Input sequence directory does not exist: {source_path}")

        clip_len = int(
            self.runtime_clip_len if window_size is None else window_size
        )
        effective_drop = (
            self.runtime_drop_first_frames
            if drop_first_frames is None
            else int(drop_first_frames)
        )
        if clip_len != self.runtime_clip_len:
            raise ValueError(
                "window_size does not match this runtime compatibility key: "
                f"{clip_len} != {self.runtime_clip_len}. Recreate SystemModelRuntime "
                "with runtime_clip_len set to the requested value."
            )
        if effective_drop != self.runtime_drop_first_frames:
            raise ValueError(
                "drop_first_frames does not match this runtime compatibility key: "
                f"{effective_drop} != {self.runtime_drop_first_frames}. Recreate "
                "SystemModelRuntime with runtime_drop_first_frames set to the requested value."
            )
        dataset = build_window_dataset(
            source=source_path,
            input_type=self.bundle.input_type,
            mode=self.bundle.mode,
            channels=self.bundle.channels,
            data_config=self.bundle.data,
            window_size=clip_len,
            stride=int(stride),
            drop_first_frames=effective_drop,
        )
        features = self.adapter.encode_dataset(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        embeddings = features.detach().float().cpu().numpy().astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise RuntimeError("The model produced no embeddings")
        if embeddings.shape[1] != int(self.model_record["embedding_dim"]):
            self.model_record = self._build_model_record(embedding_dim=int(embeddings.shape[1]))
        if not np.isfinite(embeddings).all():
            raise RuntimeError("The model produced NaN or Inf embeddings")

        effective_fingerprint = source_fingerprint or compute_source_fingerprint(
            source=source_path,
            input_type=self.bundle.input_type,
            mode=self.bundle.mode,
        )
        source_metadata = {
            **_sequence_metadata(source_path),
            "source_fingerprint": effective_fingerprint,
            "window_size": clip_len,
            "stride": int(stride),
            "drop_first_frames": effective_drop,
            "num_windows": len(dataset),
        }
        return EmbeddingBatch(
            embeddings=embeddings,
            windows=dataset.window_manifest(),
            source=source_path,
            source_metadata=source_metadata,
            model=self.model_record,
        )
