#!/usr/bin/env python3
"""Check bundled encoders against live-window encoding using synthetic foreground data."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mph_gait_id.database import GalleryRepository
from mph_gait_id.enrollment_sources import EnrollmentSourceLibrary, ForegroundCapture
from mph_gait_id.gallery_transfer import GalleryTransfer
from mph_gait_id.model_store import ModelStore
from mph_gait_id.realtime.preprocessing import _deterministic_sample
from mph_gait_id.runtime import SystemModelRuntime
from mph_gait_id.source_registration import RegistrationTarget, SourceRegistrationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bundle", action="append", dest="bundles")
    args = parser.parse_args()
    bundle_ids = args.bundles or ["pointnet_tmax_final24_len15_seed2", "mph_gait_final24_len15_seed2",
                                 "lidargaitpp_final24_len15_seed2"]
    torch.set_num_threads(1)
    store = ModelStore()
    profile = "person_foreground_pointcloud_v1"
    with tempfile.TemporaryDirectory(prefix="mph-source-smoke-") as raw:
        repo = GalleryRepository(Path(raw) / "gallery.sqlite3")
        library = EnrollmentSourceLibrary(repo)
        library.initialize()
        repo.upsert_person("synthetic", "Synthetic fixture")
        capture = ForegroundCapture(library, min_free_bytes=0, queue_size=32)
        try:
            rng = np.random.default_rng(12)
            for index in range(30):
                points = rng.uniform((-250, -1600, 2200), (250, 0, 2600), (1100 + index, 3)).astype(np.float32)
                capture.append("pass_001", points, frame_index=index, timestamp=index / 15.,
                               segment_start=index == 0, metadata={})
            prepared = capture.prepare({"pass_001"}, dict(session_id="synthetic-session",
                capture={"preprocessing_profile_id": profile}, passes=[]))
            with repo.connect() as con:
                prepared.attach(con, "synthetic", "Synthetic fixture", [])
            source_id = capture.source_id
            prepared.finish()
        finally:
            if not capture.closed:
                capture.abandon()
        targets = [RegistrationTarget(key, 15) for key in bundle_ids]
        service = SourceRegistrationService(library, store, profile)
        report = service.run(source_id, targets, ["pass_001"], max_windows_per_pass=1,
                             device=args.device, progress=print)
        results = []
        manifest = library.manifest(source_id)
        for target, item in zip(targets, report["targets"]):
            runtime = SystemModelRuntime(bundle_id=target.bundle_id, model_store=store,
                                         runtime_clip_len=15, device=args.device, preprocessing_profile_id=profile)
            points = np.stack([_deterministic_sample(library.load_frame(source_id, frame),
                item["num_points"]) for frame in manifest["frames"][:15]])
            expected = runtime.adapter.encode_tensor(torch.from_numpy(points))[0].numpy()
            actual = repo.load_gallery(item["model_key"])[0]["embedding"]
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
            assert item["model_key"] == runtime._build_model_record(embedding_dim=expected.size)["model_key"]
            results.append(dict(bundle_id=target.bundle_id, dimension=int(expected.size),
                                max_abs_difference=float(np.max(np.abs(actual - expected)))))
            del runtime
            gc.collect()
        # Verify that newly encoded part descriptors remain transferable without media.
        transfer = GalleryTransfer(repo, store, profile)
        archive = Path(raw) / "synthetic.mphgallery"
        transfer.export(archive)
        other = GalleryRepository(Path(raw) / "other" / "gallery.sqlite3")
        other.initialize()
        receiver = GalleryTransfer(other, store, profile)
        preview = receiver.preview(archive)
        imported = receiver.import_archive(archive, preview, {p["uid"]: p["target_id"] for p in preview["persons"]})
        assert imported["inserted"] == len(targets), imported
        assert service.run(source_id, targets, ["pass_001"], device=args.device)["embeddings_added"] == 0
        print(json.dumps(dict(status="passed", models=results, transferred=imported["inserted"]), indent=2))


if __name__ == "__main__":
    main()
