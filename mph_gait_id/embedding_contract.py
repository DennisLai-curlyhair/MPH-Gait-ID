"""Feature-definition snapshots and narrowly scoped legacy compatibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model_store import ModelBundle


LEGACY_V030_ENCODER = "5aa3fda0f77ca57c7403f4381bc189fbf5fcce8f414298453da727ceb492837a"
MIGRATION_TARGET_ENCODER = "0c2df80060d0751a0098b00d1f907d6677f24ec6fc9e3f2aa72fb5f65a0e897d"
CONTRACT_FIELDS = {"contract_version", "coordinate_adapter", "input_channels",
                   "input_config_sha256", "model_config_sha256"}


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def encoder_hash() -> str:
    root = Path(__file__).parent
    paths = [root / name for name in ("model_adapter.py", "dataio.py", "runtime.py", "model_record.py")]
    paths += sorted((root / "models").glob("*.py"))
    return hashlib.sha256(b"".join(
        path.relative_to(root).as_posix().encode() + b"\0" +
        path.read_bytes().replace(b"\r\n", b"\n") for path in paths
    )).hexdigest()


def bundle_spec(bundle: ModelBundle) -> dict[str, Any]:
    data = {k: v for k, v in bundle.data.items()
            if k not in {"batch_size", "clip_len", "drop_first_frames"}}
    return {"architecture": bundle.architecture, "channels": bundle.channels,
            "data": data, "model": bundle.model, "encoder_sha256": encoder_hash()}


def compatible_definition(source: dict[str, Any], target: dict[str, Any],
                          source_spec: Any, target_spec: dict[str, Any]) -> bool:
    if not isinstance(source_spec, dict):
        return False
    old = {k: v for k, v in source.items() if k != "bundle_id"}
    new = {k: v for k, v in target.items() if k != "bundle_id"}
    if old.get("contract_version") == 2:
        return old == new and source_spec == target_spec
    # Only v0.3.0's audited encoder is eligible, never an arbitrary old hash.
    if ("contract_version" in old or source_spec.get("encoder_sha256") != LEGACY_V030_ENCODER
            or target_spec.get("encoder_sha256") != MIGRATION_TARGET_ENCODER):
        return False
    if {k: v for k, v in source_spec.items() if k != "encoder_sha256"} != {
            k: v for k, v in target_spec.items() if k != "encoder_sha256"}:
        return False
    return old == {k: v for k, v in new.items() if k not in CONTRACT_FIELDS}
