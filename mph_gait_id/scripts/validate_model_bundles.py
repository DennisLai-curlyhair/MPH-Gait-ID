#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.model_store import ModelStore
from mph_gait_id.runtime import SystemModelRuntime


def main() -> int:
    store = ModelStore()
    results = []
    for bundle_id, bundle in store.bundles().items():
        runtime = SystemModelRuntime(bundle_id, device="auto", model_store=store)
        clip_len = int(bundle.data.get("clip_len", 15))
        points = int(bundle.data.get("num_points", 1024))
        inputs = torch.zeros(clip_len, points, 3, dtype=torch.float32)
        inputs[..., 2] = torch.linspace(1500.0, 2500.0, points)
        embedding = runtime.adapter.encode_tensor(inputs)
        finite = bool(torch.isfinite(embedding).all().item())
        results.append(
            {
                "bundle_id": bundle_id,
                "architecture": bundle.architecture,
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "shape": list(embedding.shape),
                "finite": finite,
                "device": str(runtime.adapter.device),
            }
        )
    print(json.dumps({"status": "ok", "bundles": results}, indent=2))
    return 0 if all(item["finite"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
