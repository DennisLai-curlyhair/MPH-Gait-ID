"""Transactional multi-model enrollment from consented foreground recordings."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import threading
from typing import Callable
import uuid

import numpy as np
import torch

from .database import _utc_now
from .enrollment_sources import EnrollmentSourceLibrary, _Lease, _json
from .gallery_transfer import _bundle_spec
from .model_record import build_model_record
from .model_store import ModelStore, descriptor_dimension, sha256_file
from .realtime.preprocessing import _deterministic_sample
from .runtime import SystemModelRuntime
from .services import embedding_coherence


class RegistrationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class RegistrationTarget:
    bundle_id: str
    clip_len: int


def plan_windows(frames: list[dict], pass_id: str, clip_len: int,
                 max_windows: int, max_gap_s: float) -> list[list[dict]]:
    """Non-overlapping windows within uninterrupted passes; no padding or first-30 drop."""
    if not 1 <= clip_len <= 300 or not 1 <= max_windows <= 100:
        raise ValueError("Clip length must be 1..300; maximum windows must be 1..100")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Invalid source sampling-gap limit")
    segments: list[list[dict]] = []
    current: list[dict] = []
    previous = None
    for frame in frames:
        if frame["pass_id"] != pass_id:
            previous = None
            current = []
            continue
        timestamp = float(frame["timestamp"])
        if not np.isfinite(timestamp) or int(frame["points"]) <= 0:
            raise ValueError("Invalid source timestamp or point count")
        interrupted = (previous is None or frame["segment_start"]
                       or int(frame["sensor_frame_index"]) <= int(previous["sensor_frame_index"])
                       or not 0 < timestamp - float(previous["timestamp"]) <= max_gap_s)
        if interrupted:
            current = []
            segments.append(current)
        current.append(frame)
        previous = frame
    windows = [segment[i:i + clip_len] for segment in segments
               for i in range(0, len(segment) - clip_len + 1, clip_len)]
    if len(windows) > max_windows:
        indices = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
        windows = [windows[i] for i in indices]
    return windows


class SourceRegistrationService:
    def __init__(self, library: EnrollmentSourceLibrary, store: ModelStore,
                 preprocessing_profile_id: str, *, runtime_factory=SystemModelRuntime):
        self.library, self.store = library, store
        self.repository = library.repository
        self.profile = preprocessing_profile_id
        self.runtime_factory = runtime_factory

    def _initialize(self):
        self.library.initialize()
        with self.repository.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS enrollment_source_jobs (
                job_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, created_at TEXT NOT NULL,
                status TEXT NOT NULL, report_json TEXT NOT NULL)""")

    def _owner(self, con, source_id):
        row = con.execute("""SELECT s.person_uid, p.person_id, p.display_name, p.status
            FROM enrollment_sources s
            LEFT JOIN gallery_transfer_aliases a ON a.kind='person' AND a.uid=s.person_uid
            LEFT JOIN persons p ON p.person_id=a.local_id WHERE s.source_id=?""",
            (source_id,)).fetchone()
        if row is None or row["person_id"] is None or row["status"] != "active":
            raise ValueError("Source owner is deleted or inactive. Restore the original person "
                             "through Gallery management/import first; a reused Person ID is not ownership.")
        return dict(row)

    def _coverage(self, con, source_id, model_key, person_id):
        covered = set()
        session_id = con.execute("SELECT session_id FROM enrollment_sources WHERE source_id=?",
                                 (source_id,)).fetchone()[0]
        # Include inactive rows: re-encoding must not silently bypass deactivation.
        for row in con.execute("""SELECT person_id, pass_id, active, metadata_json, session_id, source_fingerprint
                FROM gallery_embeddings WHERE model_key=?""", (model_key,)):
            metadata = json.loads(row["metadata_json"])
            linked = (metadata.get("source_metadata", {}).get("foreground_source_id") == source_id
                      or row["source_fingerprint"] == f"foreground-source:{source_id}"
                      or (session_id and row["session_id"] == session_id))
            if not linked:
                continue
            if row["person_id"] != person_id:
                raise ValueError("This foreground source is already associated with another person")
            covered.add(row["pass_id"] or "*")
        return covered

    @staticmethod
    def _save_report(con, report):
        con.execute("INSERT INTO enrollment_source_jobs VALUES (?, ?, ?, ?, ?)",
                    (report["job_id"], report["source_id"], report["created_at"],
                     report["status"], _json(report).decode()))

    def run(self, source_id: str, targets: list[RegistrationTarget], pass_ids: list[str], *,
            device: str = "auto", max_windows_per_pass: int = 10,
            cancel: threading.Event | None = None, progress: Callable[[str], None] | None = None):
        self._initialize()
        cancel = cancel if cancel is not None else threading.Event()
        progress = progress or (lambda _message: None)
        report = dict(job_id=str(uuid.uuid4()), source_id=source_id, created_at=_utc_now(),
                      status="preparing", targets=[], embeddings_added=0,
                      selected_pass_ids=list(pass_ids), max_windows_per_pass=max_windows_per_pass,
                      device=device, encoder_source_sha256=sha256_file(Path(__file__)))
        report["point_sampler_source_sha256"] = sha256_file(Path(_deterministic_sample.__code__.co_filename))
        committed = False

        def check_cancel():
            if cancel.is_set():
                raise RegistrationCancelled("Cancelled; no new Gallery embeddings were saved")

        try:
            with _Lease(self.library.root):
                check_cancel()
                if not targets or not pass_ids or len(targets) > 32:
                    raise ValueError("Select passes and 1..32 model targets")
                if len(set(targets)) != len(targets) or len(set(pass_ids)) != len(pass_ids):
                    raise ValueError("Duplicate target or pass selection")
                manifest = self.library.manifest(source_id)
                manifest_hash = hashlib.sha256(_json(manifest)).hexdigest()
                if (manifest.get("coordinate_convention") != "camera_X_right_Y_down_Z_forward"
                        or manifest.get("units") != "mm" or manifest.get("centered") is not False
                        or manifest.get("scale_normalized") is not False
                        or manifest.get("storage_stage") != "foreground_before_sampling"):
                    raise ValueError("Expected uncentered, unsampled foreground camera XYZ in mm")
                capture = manifest["metadata"]["capture"]
                if capture.get("preprocessing_profile_id") != self.profile:
                    raise ValueError("Source preprocessing profile does not match this application; "
                                     "stored foreground cannot be relabeled as a new preprocessing version")
                available = {f["pass_id"] for f in manifest["frames"]}
                if not set(pass_ids) <= available:
                    raise ValueError("A selected pass is not present in the source")
                gap = float(capture.get("filter_parameters", {}).get("max_valid_frame_gap_s", 1.0))
                with self.repository.connect() as con:
                    owner = self._owner(con, source_id)
                report.update(person_id=owner["person_id"], person_uid=owner["person_uid"],
                              manifest_sha256=manifest_hash, preprocessing_profile_id=self.profile,
                              stride_policy="clip_len", drop_saved_foreground_frames=0)
                pending = []
                snapshots = []
                # Validate every target before allocating any model or writing embeddings.
                for target in targets:
                    check_cancel()
                    bundle = self.store.get(target.bundle_id)
                    if bundle.input_type != "pointcloud":
                        raise ValueError("Only point-cloud targets can consume foreground recordings")
                    adapter = bundle.data.get("coordinate_adapter", "none")
                    if adapter not in {"none", "identity", "kinect_xyz_mm_to_forward_lateral_height_m_v1"}:
                        raise ValueError(f"Target expects a different source coordinate convention: {adapter}")
                    n = int(bundle.data.get("num_points", 1024))
                    if not 1 <= n <= 65536 or target.clip_len * n > 2_000_000:
                        raise ValueError("Target point count/window exceeds the source-registration limit")
                    verification = self.store.verify(target.bundle_id)
                    if not verification["valid"]:
                        raise ValueError(f'{target.bundle_id}: {verification["errors"]}')
                    record = build_model_record(bundle, self.store, target.clip_len,
                                                int(bundle.data.get("drop_first_frames", 30)), self.profile,
                                                embedding_dim=descriptor_dimension(bundle))
                    spec = _bundle_spec(bundle)
                    snapshots.append((target, _json(spec), record))
                    item = dict(bundle_id=target.bundle_id, model_key=record["model_key"],
                                clip_len=target.clip_len, num_points=n, status="pending",
                                checkpoint_sha256=bundle.checkpoint_sha256, passes=[], embeddings_added=0)
                    report["targets"].append(item)
                    with self.repository.connect() as con:
                        saved = con.execute("SELECT spec_json FROM gallery_transfer_model_specs WHERE model_key=?",
                                            (record["model_key"],)).fetchone()
                        if saved and json.loads(saved[0]) != spec:
                            raise ValueError("Target encoding contract differs from the existing Gallery")
                        covered = self._coverage(con, source_id, record["model_key"], owner["person_id"])
                    plans = []
                    for pass_id in pass_ids:
                        windows = plan_windows(manifest["frames"], pass_id, target.clip_len,
                                               max_windows_per_pass, gap)
                        state = "already_registered" if pass_id in covered or "*" in covered else (
                            "ready" if windows else "too_short")
                        item["passes"].append(dict(pass_id=pass_id, status=state, windows=len(windows)))
                        if state == "ready":
                            plans.append((pass_id, windows))
                    pending.append((target, record, item, plans))

                encoded = []
                consumed_frames = {}
                pass_details = {p["pass_id"]: p for p in manifest["metadata"].get("passes", [])}
                for target_index, (target, record, item, plans) in enumerate(pending, 1):
                    check_cancel()
                    if not plans:
                        item["status"] = "skipped"
                        progress(f'[{target_index}/{len(targets)}] {target.bundle_id}: skipped')
                        continue
                    progress(f'[{target_index}/{len(targets)}] Loading {target.bundle_id}, T={target.clip_len}')
                    runtime = None
                    try:
                        gc.collect()
                        runtime = self.runtime_factory(bundle_id=target.bundle_id, device=device,
                            model_store=self.store, runtime_clip_len=target.clip_len,
                            preprocessing_profile_id=self.profile)
                        bundle = self.store.get(target.bundle_id)
                        initial_record = build_model_record(bundle, self.store, target.clip_len,
                            int(bundle.data.get("drop_first_frames", 30)), self.profile)
                        if runtime.model_record != initial_record:
                            raise ValueError("Runtime and planned model contracts differ")
                        total = sum(len(windows) for _pass, windows in plans)
                        done = 0
                        for pass_id, windows in plans:
                            vectors, metadata = [], []
                            for index, window in enumerate(windows):
                                check_cancel()
                                sampled = []
                                for frame in window:
                                    check_cancel()
                                    points = self.library.load_frame(source_id, frame)
                                    consumed_frames[frame["file"]] = frame["sha256"]
                                    sampled.append(_deterministic_sample(points, item["num_points"]))
                                values = runtime.adapter.encode_tensor(torch.from_numpy(np.stack(sampled)))
                                matrix = values.numpy().astype(np.float32)
                                if (matrix.shape != (1, record["embedding_dim"])
                                        or not np.isfinite(matrix).all()
                                        or np.linalg.norm(matrix[0]) < 1e-8):
                                    raise ValueError("Invalid/zero embedding or unexpected descriptor dimension")
                                vectors.append(matrix[0] / np.linalg.norm(matrix[0]))
                                timing = np.array([f["timestamp"] for f in window], dtype=float)
                                details = pass_details.get(pass_id, {})
                                metadata.append(dict(
                                    window=dict(window_index=index, pass_id=pass_id,
                                        requested_direction=details.get("requested_direction", ""),
                                        observed_direction=details.get("observed_direction", ""),
                                        frame_indices=[f["sensor_frame_index"] for f in window],
                                        start_frame_number=window[0]["sensor_frame_index"],
                                        end_frame_number=window[-1]["sensor_frame_index"],
                                        captured_at=float(timing[-1]), window_duration_s=float(timing[-1] - timing[0])),
                                    source_metadata=dict(foreground_source_id=source_id,
                                        source_registration_job_id=report["job_id"],
                                        manifest_sha256=manifest_hash,
                                        preprocessing_profile_id=self.profile,
                                        realtime_enrollment=dict(session_id=manifest["metadata"]["session_id"]),
                                        drop_saved_foreground_frames=0), model_key=record["model_key"]))
                                done += 1
                                progress(f'[{target_index}/{len(targets)}] {target.bundle_id}: {done}/{total} windows')
                            encoded.append((record, item, pass_id, np.stack(vectors), metadata))
                        item["status"] = "encoded"
                    finally:
                        # Do not add re-encoding models to the application's persistent runtime cache.
                        del runtime
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                check_cancel()
                if hashlib.sha256(_json(self.library.manifest(source_id))).hexdigest() != manifest_hash:
                    raise ValueError("Source manifest changed during registration")
                for filename, digest in consumed_frames.items():
                    check_cancel()
                    path = self.library.record_dir(source_id) / filename
                    if path.is_symlink() or sha256_file(path) != digest:
                        raise ValueError("Source frame changed during registration")
                for target, snapshot, _record in snapshots:
                    self.store.refresh()
                    bundle = self.store.get(target.bundle_id)
                    if (_json(_bundle_spec(bundle)) != snapshot or build_model_record(
                            bundle, self.store, target.clip_len, int(bundle.data.get("drop_first_frames", 30)),
                            self.profile, embedding_dim=descriptor_dimension(bundle)) != _record):
                        raise ValueError("Target bundle changed during registration")
                    if not self.store.verify(target.bundle_id)["valid"]:
                        raise ValueError("Checkpoint changed during registration")
                progress("Committing all new model galleries...")
                with self.repository.connect() as con:
                    con.execute("BEGIN IMMEDIATE")
                    if self._owner(con, source_id) != owner:
                        raise ValueError("Source owner changed during registration; review and retry")
                    for record, item, pass_id, matrix, metadata in encoded:
                        check_cancel()
                        covered = self._coverage(con, source_id, record["model_key"], owner["person_id"])
                        if pass_id in covered or "*" in covered:
                            raise ValueError("Gallery changed during registration; refresh and retry")
                        self.repository.upsert_model(record, connection=con)
                        spec_json = next(snapshot.decode() for _target, snapshot, model in snapshots
                                         if model["model_key"] == record["model_key"])
                        saved = con.execute("SELECT spec_json FROM gallery_transfer_model_specs WHERE model_key=?",
                                            (record["model_key"],)).fetchone()
                        if saved and json.loads(saved[0]) != json.loads(spec_json):
                            raise ValueError("Gallery encoding contract changed; refresh and retry")
                        con.execute("INSERT OR IGNORE INTO gallery_transfer_model_specs VALUES (?,?)",
                                    (record["model_key"], spec_json))
                        self.repository.add_embeddings(owner["person_id"], record["model_key"], matrix,
                            self.library.record_dir(source_id), f"foreground-source:{source_id}",
                            "enrollment_source_library", embedding_coherence(matrix), metadata, connection=con)
                        item["embeddings_added"] += len(matrix)
                        report["embeddings_added"] += len(matrix)
                    check_cancel()
                    report["status"] = "completed"
                    for item in report["targets"]:
                        if item["status"] == "encoded":
                            item["status"] = "registered"
                        for pass_result in item["passes"]:
                            if pass_result["status"] == "ready":
                                pass_result["status"] = "registered"
                    self._save_report(con, report)
                committed = True
                return report
        except Exception as exc:
            if committed:
                report["warning"] = f"Gallery committed, but cleanup failed: {exc}"
                return report
            report.update(status="cancelled" if isinstance(exc, RegistrationCancelled) else "failed",
                          embeddings_added=0, error=str(exc))
            for item in report["targets"]:
                item["embeddings_added"] = 0
                if item["status"] in {"registered", "encoded"}:
                    item["status"] = "not_saved"
                for pass_result in item["passes"]:
                    if pass_result["status"] in {"ready", "registered"}:
                        pass_result["status"] = "not_saved"
            # Diagnostic history is independent of the rolled-back Gallery transaction.
            try:
                with self.repository.connect() as con:
                    self._save_report(con, report)
            except Exception:
                pass
            if isinstance(exc, RegistrationCancelled):
                return report
            raise
