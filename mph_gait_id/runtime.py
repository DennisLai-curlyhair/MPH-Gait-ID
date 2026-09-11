from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataio import build_window_dataset
from .identity import parse_sequence_metadata
from .model_adapter import ModelAdapter
from .model_record import build_model_record
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

    def _build_model_record(self, embedding_dim: int | None = None) -> dict[str, Any]:
        return build_model_record(
            self.bundle, self.model_store, self.runtime_clip_len,
            self.runtime_drop_first_frames, self.preprocessing_profile_id,
            embedding_dim=embedding_dim,
        )

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
