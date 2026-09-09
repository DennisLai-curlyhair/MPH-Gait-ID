"""Model metadata shared by live inference and portable Gallery transfer."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .model_store import ModelBundle, ModelStore


def build_model_record(
    bundle: ModelBundle,
    store: ModelStore,
    clip_len: int,
    drop_first_frames: int,
    preprocessing_profile_id: str,
    embedding_dim: int | None = None,
) -> dict[str, Any]:
    dimension = int(bundle.model.get("embedding_dim", 256))
    if bundle.architecture == "lidargaitpp_official":
        dimension *= int(bundle.model.get("parts_num", 1))
    dimension = int(embedding_dim or dimension)
    try:
        checkpoint = str(bundle.checkpoint.relative_to(store.root.parent))
    except ValueError:
        checkpoint = str(bundle.checkpoint)
    compatibility = {
        "bundle_id": bundle.bundle_id,
        "method_key": bundle.method_key,
        "architecture": bundle.architecture,
        "input_type": bundle.input_type,
        "input_mode": bundle.mode,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "clip_len": int(clip_len),
        "drop_first_frames": int(drop_first_frames),
        "embedding_dim": dimension,
        "num_points": bundle.data.get("num_points"),
        "point_normalization": bundle.model.get("point_normalization"),
        "preprocessing_profile_id": preprocessing_profile_id,
    }
    digest = hashlib.sha256(
        json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "model_key": f"{bundle.bundle_id}_{digest[:16]}",
        "bundle_id": bundle.bundle_id,
        "method_key": bundle.method_key,
        "display_name": bundle.display_name,
        "architecture": bundle.architecture,
        "input_type": bundle.input_type,
        "input_mode": bundle.mode,
        "fold": bundle.fold,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "embedding_dim": dimension,
        "compatibility": compatibility,
        "research_fold_checkpoint": bundle.fold >= 0,
    }
