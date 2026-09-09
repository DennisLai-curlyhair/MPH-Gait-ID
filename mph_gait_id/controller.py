from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import DEFAULT_CONFIG_PATH, load_config, nested, resolve_system_path
from .database import GalleryRepository
from .gallery_lifecycle import GalleryLifecycle
from .gallery_transfer import GalleryTransfer
from .model_store import ModelBundle, ModelStore
from .runtime import EmbeddingBatch, SystemModelRuntime
from .services import RecognitionService, RegistrationService
from .source_adapter import PreparedSource, SourcePreparer, discover_sequence_folders


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class OperationOutcome:
    operation: str
    result: dict[str, Any]
    prepared_source: PreparedSource
    elapsed_seconds: float


class GaitApplicationController:
    """Application boundary shared by the desktop UI and future sensor adapters."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        database_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.config = load_config(config_path)
        configured_database = database_path or nested(
            self.config, "storage", "database", "../data/gallery.sqlite3"
        )
        configured_output = output_root or nested(
            self.config, "storage", "output_root", "../outputs"
        )
        bundle_root = nested(self.config, "storage", "model_bundles", "model_bundles")
        folder_fps = float(nested(self.config, "ui", "folder_fps", 10.0))

        self.repository = GalleryRepository(resolve_system_path(configured_database))
        self.repository.initialize()
        self.output_root = resolve_system_path(configured_output)
        self.source_preparer = SourcePreparer(folder_fps=folder_fps)
        self.model_store = ModelStore(resolve_system_path(bundle_root))
        for bundle in self.model_store.bundles().values():
            self.repository.synchronize_bundle_display_name(
                bundle_id=bundle.bundle_id,
                checkpoint_sha256=bundle.checkpoint_sha256,
                display_name=bundle.display_name,
            )
        self._runtime_cache: dict[
            tuple[str, str, int, int, str], SystemModelRuntime
        ] = {}
        self._runtime_lock = threading.RLock()

    def available_bundles(self, refresh: bool = False) -> dict[str, ModelBundle]:
        return self.model_store.refresh() if refresh else self.model_store.bundles()

    def import_checkpoint(
        self,
        checkpoint_path: str | Path,
        display_name: str,
        template_bundle_id: str | None = None,
    ) -> ModelBundle:
        return self.model_store.import_checkpoint(
            checkpoint_path=checkpoint_path,
            display_name=display_name,
            template_bundle_id=template_bundle_id,
        )

    def database_summary(
        self,
        bundle_id: str | None = None,
        clip_len: int | None = None,
    ) -> dict[str, Any]:
        if bundle_id is None:
            return self.repository.summary()
        self.model_store.get(bundle_id)
        model_keys = self._compatible_model_keys(bundle_id, clip_len=clip_len)
        result = self.repository.summary(model_keys=model_keys)
        result["bundle_id"] = bundle_id
        result["clip_len"] = clip_len
        return result

    def list_persons(
        self,
        bundle_id: str | None = None,
        clip_len: int | None = None,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        if bundle_id is None:
            return self.repository.list_persons(include_inactive=include_inactive)
        self.model_store.get(bundle_id)
        model_keys = self._compatible_model_keys(bundle_id, clip_len=clip_len)
        return self.repository.list_persons(
            model_keys=model_keys,
            include_inactive=include_inactive,
        )

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        return self.repository.get_person(person_id)

    def preview_person_edit(self, person_id: str) -> dict[str, Any]:
        return GalleryLifecycle(self.repository).preview(person_id)

    def preview_fragment_delete(
        self, bundle_id: str, person_id: str, fragment: dict[str, Any], clip_len: int,
    ) -> dict[str, Any]:
        if (fragment.get("person_id") != person_id
                or fragment.get("model_key") not in self._compatible_model_keys(bundle_id, clip_len)):
            raise ValueError("Selected fragment does not belong to this person/model scope")
        selector = {key: fragment[key] for key in (
            "model_key", "source_path", "session_id", "pass_id", "direction"
        )}
        return GalleryLifecycle(self.repository).preview(person_id, selector)

    def apply_person_edit(self, preview: dict[str, Any], action: str, new_name: str = "") -> dict:
        return GalleryLifecycle(self.repository).apply(preview, action, new_name)

    def list_gallery_sources(
        self,
        bundle_id: str,
        person_id: str,
    ) -> list[dict[str, Any]]:
        self.model_store.get(bundle_id)
        model_keys = self._compatible_model_keys(bundle_id)
        return self.repository.list_gallery_sources(person_id, model_keys)

    def list_gallery_passes(
        self,
        bundle_id: str,
        person_id: str,
        clip_len: int | None = None,
        include_inactive: bool = True,
    ) -> list[dict[str, Any]]:
        self.model_store.get(bundle_id)
        model_keys = self._compatible_model_keys(bundle_id, clip_len=clip_len)
        return self.repository.list_gallery_passes(
            person_id,
            model_keys,
            include_inactive=include_inactive,
        )

    def set_gallery_pass_active(
        self,
        bundle_id: str,
        person_id: str,
        model_key: str,
        session_id: str,
        pass_id: str,
        active: bool,
    ) -> int:
        self.model_store.get(bundle_id)
        allowed_model_keys = set(self._compatible_model_keys(bundle_id))
        if model_key not in allowed_model_keys:
            raise ValueError("The selected pass does not belong to this model bundle")
        return self.repository.set_pass_embeddings_active(
            person_id=person_id,
            model_key=model_key,
            session_id=session_id,
            pass_id=pass_id,
            active=active,
        )

    def deactivate_gallery_source(
        self,
        bundle_id: str,
        person_id: str,
        model_key: str,
        source_path: str | Path,
    ) -> int:
        self.model_store.get(bundle_id)
        allowed_model_keys = set(self._compatible_model_keys(bundle_id))
        if model_key not in allowed_model_keys:
            raise ValueError(
                "The selected source does not belong to the active model bundle"
            )
        return self.repository.deactivate_source_embeddings(
            person_id=person_id,
            model_key=model_key,
            source_path=source_path,
        )

    def deactivate_person_gallery(
        self,
        bundle_id: str,
        person_id: str,
    ) -> int:
        self.model_store.get(bundle_id)
        model_keys = self._compatible_model_keys(bundle_id)
        return self.repository.deactivate_person_embeddings(person_id, model_keys)

    def gallery_transfer(self) -> GalleryTransfer:
        return GalleryTransfer(self.repository, self.model_store, self.processing_version_id())

    def export_gallery(self, path: str | Path) -> dict[str, Any]:
        return self.gallery_transfer().export(path)

    def preview_gallery_import(self, path: str | Path) -> dict[str, Any]:
        return self.gallery_transfer().preview(path)

    def import_gallery(
        self, path: str | Path, preview: dict[str, Any],
        person_map: dict[str, str | None],
    ) -> dict[str, Any]:
        return self.gallery_transfer().import_archive(path, preview, person_map)

    def processing_version_id(self) -> str:
        return str(
            nested(
                self.config,
                "realtime",
                "processing_version_id",
                "person_foreground_pointcloud_v1",
            )
        )

    def processing_version_name(self, locale: str = "zh_TW") -> str:
        key = "processing_version_name_zh" if locale == "zh_TW" else "processing_version_name_en"
        fallback = "人體前景點雲 v1" if locale == "zh_TW" else "Person foreground point cloud v1"
        return str(nested(self.config, "realtime", key, fallback))

    def _compatible_model_keys(
        self,
        bundle_id: str,
        clip_len: int | None = None,
    ) -> list[str]:
        return self.repository.model_keys_for_bundle(
            bundle_id,
            clip_len=clip_len,
            preprocessing_profile_id=self.processing_version_id(),
        )

    def provisional_threshold(
        self,
        bundle_id: str,
        clip_len: int | None = None,
    ) -> float:
        bundle = self.model_store.get(bundle_id)
        configured = nested(
            self.config,
            "recognition",
            "provisional_thresholds",
            {},
        )
        method_config = (
            configured.get(bundle.method_key, {})
            if isinstance(configured, dict)
            else {}
        )
        effective_clip_len = int(
            clip_len
            if clip_len is not None
            else bundle.data.get("clip_len", 15)
        )
        if isinstance(method_config, dict):
            value = method_config.get(str(effective_clip_len))
            if value is not None:
                return float(value)
        elif method_config is not None:
            return float(method_config)
        return float(
            nested(
                self.config,
                "recognition",
                "provisional_threshold",
                0.55,
            )
        )

    def discover_sequence_sources(
        self,
        root: str | Path,
        bundle_id: str,
    ) -> list[Path]:
        bundle = self.model_store.get(bundle_id)
        return discover_sequence_folders(root, bundle)

    def enroll(
        self,
        source: str | Path,
        bundle_id: str,
        person_id: str,
        display_name: str,
        note: str = "",
        device: str = "auto",
        batch_size: int | None = None,
        num_workers: int | None = None,
        stride: int | None = None,
        min_clips: int | None = None,
        max_embeddings: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> OperationOutcome:
        return self.enroll_many(
            sources=[source],
            bundle_id=bundle_id,
            person_id=person_id,
            display_name=display_name,
            note=note,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            stride=stride,
            min_clips=min_clips,
            max_embeddings_per_source=max_embeddings,
            progress=progress,
        )

    def enroll_many(
        self,
        sources: Sequence[str | Path],
        bundle_id: str,
        person_id: str,
        display_name: str,
        note: str = "",
        device: str = "auto",
        batch_size: int | None = None,
        num_workers: int | None = None,
        stride: int | None = None,
        min_clips: int | None = None,
        max_embeddings_per_source: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> OperationOutcome:
        source_items = list(sources)
        if not source_items:
            raise ValueError("At least one enrollment source is required")
        person_id = person_id.strip()
        display_name = display_name.strip()
        if not person_id or not display_name:
            raise ValueError("person_id and display_name must not be empty")
        existing_person = self.repository.get_person(person_id)
        if (
            existing_person is not None
            and str(existing_person["display_name"]) != display_name
        ):
            raise ValueError(
                f"Person ID {person_id} 已登記為「{existing_person['display_name']}」。"
                "顯示名稱代表人物本身；服裝條件請保留在來源資料，不要加入名稱。"
            )
        started = time.perf_counter()
        callback = progress or (lambda _message: None)
        callback("載入本機模型 Bundle 與前處理設定")
        runtime = self._runtime(bundle_id, device, batch_size, num_workers)

        prepared_sources: list[PreparedSource] = []
        batches: list[EmbeddingBatch] = []
        for index, source in enumerate(source_items, start=1):
            callback(f"準備註冊來源 {index}/{len(source_items)}：{Path(source).name}")
            prepared, batch = self._extract(runtime, source, stride)
            prepared_sources.append(prepared)
            batches.append(batch)
            callback(
                f"來源 {index}/{len(source_items)} 已取得 {batch.embeddings.shape[0]} 個 clips"
            )

        minimum = int(
            min_clips if min_clips is not None else nested(self.config, "registration", "min_clips", 5)
        )
        maximum = int(
            max_embeddings_per_source
            if max_embeddings_per_source is not None
            else nested(self.config, "registration", "max_embeddings", 10)
        )
        callback(
            f"所有 {len(batches)} 個來源抽取完成，正在依來源寫入 Gallery"
        )
        result = RegistrationService(self.repository).enroll_many(
            batches=batches,
            person_id=person_id,
            display_name=display_name,
            note=note,
            min_embeddings=minimum,
            max_embeddings_per_source=maximum,
        )
        result["bundle_id"] = runtime.bundle.bundle_id
        result["bundle_display_name"] = runtime.bundle.display_name
        result["source_adapters"] = [item.metadata() for item in prepared_sources]
        return self._finish("enroll", result, prepared_sources[0], started)

    def recognize(
        self,
        source: str | Path,
        bundle_id: str,
        threshold: float,
        min_margin: float,
        device: str = "auto",
        batch_size: int | None = None,
        num_workers: int | None = None,
        stride: int | None = None,
        top_k_per_identity: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> OperationOutcome:
        started = time.perf_counter()
        callback = progress or (lambda _message: None)
        callback("載入本機模型 Bundle 與相容 Gallery")
        runtime = self._runtime(bundle_id, device, batch_size, num_workers)
        callback("準備資料來源")
        prepared, batch = self._extract(runtime, source, stride)
        callback(f"已取得 {batch.embeddings.shape[0]} 個 clips，正在進行身分比對")

        top_k = int(
            top_k_per_identity
            if top_k_per_identity is not None
            else nested(self.config, "recognition", "top_k_per_identity", 3)
        )
        result = RecognitionService(self.repository).recognize(
            batch=batch,
            top_k_per_identity=top_k,
            threshold=float(threshold),
            min_margin=float(min_margin),
        )
        result["bundle_id"] = runtime.bundle.bundle_id
        result["bundle_display_name"] = runtime.bundle.display_name
        return self._finish("recognize", result, prepared, started)

    def _runtime(
        self,
        bundle_id: str,
        device: str,
        batch_size: int | None,
        num_workers: int | None,
    ) -> SystemModelRuntime:
        self.model_store.get(bundle_id)
        effective_batch = int(
            batch_size if batch_size is not None else nested(self.config, "runtime", "batch_size", 16)
        )
        effective_workers = int(
            num_workers if num_workers is not None else nested(self.config, "runtime", "num_workers", 0)
        )
        profile_id = self.processing_version_id()
        key = (
            bundle_id,
            str(device),
            effective_batch,
            effective_workers,
            profile_id,
        )
        with self._runtime_lock:
            runtime = self._runtime_cache.get(key)
            if runtime is None:
                runtime = SystemModelRuntime(
                    bundle_id=bundle_id,
                    device=device,
                    batch_size=effective_batch,
                    num_workers=effective_workers,
                    model_store=self.model_store,
                    preprocessing_profile_id=profile_id,
                )
                self._runtime_cache[key] = runtime
        return runtime

    def _extract(
        self,
        runtime: SystemModelRuntime,
        source: str | Path,
        stride: int | None,
    ) -> tuple[PreparedSource, EmbeddingBatch]:
        prepared = self.source_preparer.prepare(source, runtime.bundle)
        effective_stride = int(
            stride if stride is not None else nested(self.config, "folder_input", "stride", 5)
        )
        batch = runtime.extract_folder(
            source=prepared.inference_path,
            stride=effective_stride,
            source_fingerprint=prepared.source_fingerprint,
        )
        batch.source = prepared.original_path
        batch.source_metadata["source_adapter"] = prepared.metadata()
        return prepared, batch

    def _finish(
        self,
        operation: str,
        result: dict[str, Any],
        prepared: PreparedSource,
        started: float,
    ) -> OperationOutcome:
        elapsed = time.perf_counter() - started
        result["elapsed_seconds"] = elapsed
        result["source_adapter"] = prepared.metadata()
        self._save_result(operation, result)
        return OperationOutcome(operation, result, prepared, elapsed)

    def _save_result(self, operation: str, result: dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = self.output_root / f"{timestamp}_ui_{operation}"
        output_dir.mkdir(parents=True, exist_ok=False)
        target = output_dir / "result.json"
        result["result_path"] = str(target)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target
