from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .database import GalleryRepository
from .runtime import EmbeddingBatch
from .source_fingerprint import compute_source_fingerprint


def _normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _even_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum).round().astype(np.int64)


def embedding_coherence(embeddings: np.ndarray) -> float:
    values = _normalize(embeddings)
    centroid = _normalize(values.mean(axis=0, keepdims=True))
    return float(np.mean(values @ centroid.T))


def _batch_source_fingerprint(batch: EmbeddingBatch) -> str:
    adapter = batch.source_metadata.get("source_adapter", {})
    fingerprint = adapter.get("source_fingerprint") if isinstance(adapter, dict) else None
    fingerprint = fingerprint or batch.source_metadata.get("source_fingerprint")
    if fingerprint:
        return str(fingerprint)
    return compute_source_fingerprint(
        source=batch.source,
        input_type=str(batch.model["input_type"]),
        mode=str(batch.model["input_mode"]),
    )


class RegistrationService:
    def __init__(self, repository: GalleryRepository) -> None:
        self.repository = repository

    def enroll(
        self,
        batch: EmbeddingBatch,
        person_id: str,
        display_name: str,
        note: str = "",
        min_embeddings: int = 5,
        max_embeddings: int = 10,
        allow_duplicate_source: bool = False,
    ) -> dict[str, Any]:
        return self.enroll_many(
            batches=[batch],
            person_id=person_id,
            display_name=display_name,
            note=note,
            min_embeddings=min_embeddings,
            max_embeddings_per_source=max_embeddings,
            allow_duplicate_source=allow_duplicate_source,
        )

    def enroll_many(
        self,
        batches: Sequence[EmbeddingBatch],
        person_id: str,
        display_name: str,
        note: str = "",
        min_embeddings: int = 5,
        max_embeddings_per_source: int = 10,
        allow_duplicate_source: bool = False,
    ) -> dict[str, Any]:
        """Enroll multiple sequences while retaining source-level provenance."""

        self.repository.initialize()
        items = list(batches)
        if not items:
            raise ValueError("At least one enrollment source is required")
        if max_embeddings_per_source < min_embeddings:
            raise ValueError(
                "max_embeddings_per_source must be greater than or equal to min_embeddings"
            )

        model_key = str(items[0].model["model_key"])
        seen_sources: set[str] = set()
        seen_fingerprints: dict[str, str] = {}
        selected_sources: list[dict[str, Any]] = []
        for batch in items:
            if str(batch.model["model_key"]) != model_key:
                raise ValueError("All enrollment sources must use the same model bundle")
            if batch.embeddings.shape[0] < min_embeddings:
                raise ValueError(
                    f"Insufficient clips for {batch.source}: got {batch.embeddings.shape[0]}, "
                    f"need at least {min_embeddings}"
                )
            source = str(Path(batch.source).expanduser().resolve())
            fingerprint = _batch_source_fingerprint(batch)
            if source in seen_sources:
                raise ValueError(f"The same source was selected more than once: {source}")
            if fingerprint in seen_fingerprints:
                raise ValueError(
                    "兩個待註冊來源的路徑不同，但點雲 frame 內容完全相同：\n"
                    f"{seen_fingerprints[fingerprint]}\n{source}"
                )
            seen_sources.add(source)
            seen_fingerprints[fingerprint] = source
            if not allow_duplicate_source:
                duplicate = self.repository.find_enrolled_source(
                    model_key=model_key,
                    source_path=batch.source,
                    source_fingerprint=fingerprint,
                )
                if duplicate is not None:
                    raise ValueError(
                        "此來源已在所選模型的 Gallery 中註冊。即使路徑或資料夾名稱不同，"
                        "內容指紋相同也會被視為重複，以避免資料洩漏。\n\n"
                        f"目前來源：{source}\n"
                        f"既有來源：{duplicate['source_path']}\n"
                        f"既有人物：{duplicate['person_id']}"
                    )

            selected = _even_indices(
                batch.embeddings.shape[0], max_embeddings_per_source
            )
            embeddings = _normalize(batch.embeddings[selected])
            if not np.isfinite(embeddings).all():
                raise ValueError(f"Embeddings contain NaN or Inf for source: {source}")
            selected_sources.append(
                {
                    "batch": batch,
                    "source": source,
                    "source_fingerprint": fingerprint,
                    "selected": selected,
                    "embeddings": embeddings,
                    "quality": embedding_coherence(embeddings),
                }
            )

        existing_person = self.repository.get_person(person_id)
        if (
            existing_person is not None
            and str(existing_person["display_name"]) != display_name
        ):
            raise ValueError(
                f"Person ID {person_id} 已登記為「{existing_person['display_name']}」。"
                "若為同一人，請到 Gallery 管理 → 修改人物名稱，再重新註冊。"
                "同一人物不可因服裝不同而使用不同顯示名稱。"
            )

        # All validation and feature extraction completes before a single atomic
        # Gallery transaction begins. A failure in any source rolls everything back.
        all_embedding_ids: list[int] = []
        source_results: list[dict[str, Any]] = []
        selected_matrices: list[np.ndarray] = []
        with self.repository.connect() as connection:
            self.repository.upsert_person(
                person_id, display_name, note, connection=connection
            )
            self.repository.upsert_model(items[0].model, connection=connection)
            for source_index, item in enumerate(selected_sources):
                batch = item["batch"]
                selected = item["selected"]
                embeddings = item["embeddings"]
                quality = float(item["quality"])
                adapter = batch.source_metadata.get("source_adapter", {})
                source_type = str(adapter.get("source_kind", "folder"))
                metadata = [
                    {
                        "window": batch.windows[window_index],
                        "source_metadata": batch.source_metadata,
                        "model_key": model_key,
                        "source_fingerprint": item["source_fingerprint"],
                        "enrollment_source_index": source_index,
                    }
                    for window_index in selected.tolist()
                ]
                embedding_ids = self.repository.add_embeddings(
                    person_id=person_id,
                    model_key=model_key,
                    embeddings=embeddings,
                    source_path=batch.source,
                    source_fingerprint=item["source_fingerprint"],
                    source_type=source_type,
                    quality_score=quality,
                    metadata=metadata,
                    connection=connection,
                )
                all_embedding_ids.extend(embedding_ids)
                selected_matrices.append(embeddings)
                source_results.append(
                    {
                        "source": item["source"],
                        "source_fingerprint": item["source_fingerprint"],
                        "available_clips": int(batch.embeddings.shape[0]),
                        "stored_embeddings": len(embedding_ids),
                        "embedding_ids": embedding_ids,
                        "embedding_coherence": quality,
                        "source_metadata": batch.source_metadata,
                    }
                )

        combined = np.concatenate(selected_matrices, axis=0)
        first = source_results[0]
        return {
            "operation": "enroll",
            "person_id": person_id,
            "display_name": display_name,
            "model_key": model_key,
            "source": first["source"],
            "sources": [item["source"] for item in source_results],
            "source_fingerprints": [
                item["source_fingerprint"] for item in source_results
            ],
            "source_count": len(source_results),
            "available_clips": sum(item["available_clips"] for item in source_results),
            "stored_embeddings": len(all_embedding_ids),
            "max_embeddings_per_source": int(max_embeddings_per_source),
            "embedding_ids": all_embedding_ids,
            "embedding_coherence": embedding_coherence(combined),
            "source_metadata": first["source_metadata"],
            "source_results": source_results,
        }


class RecognitionService:
    def __init__(self, repository: GalleryRepository) -> None:
        self.repository = repository

    def recognize(
        self,
        batch: EmbeddingBatch,
        top_k_per_identity: int = 3,
        threshold: float | None = None,
        min_margin: float = 0.03,
        allow_source_overlap: bool = False,
    ) -> dict[str, Any]:
        self.repository.initialize()
        model_key = str(batch.model["model_key"])
        gallery = self.repository.load_gallery(model_key)
        if not gallery:
            raise ValueError(
                "No compatible gallery embeddings exist for this exact model/checkpoint"
            )
        source = str(Path(batch.source).expanduser().resolve())
        source_fingerprint = _batch_source_fingerprint(batch)
        overlapping = sorted(
            {
                item["source_path"]
                for item in gallery
                if item["source_path"] == source
                or (
                    item.get("source_fingerprint")
                    and item["source_fingerprint"] == source_fingerprint
                )
            }
        )
        if overlapping and not allow_source_overlap:
            raise ValueError(
                "Probe 來源與 Gallery 中的既有來源具有相同路徑或內容指紋。"
                "即使資料夾被複製或改名，仍不可用同一段序列同時註冊與辨識，"
                "以避免資料洩漏。\n\nGallery 來源："
                + "\n".join(overlapping)
            )

        probes = _normalize(batch.embeddings)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in gallery:
            grouped[item["person_id"]].append(item)

        identity_scores: dict[str, np.ndarray] = {}
        identity_names: dict[str, str] = {}
        identity_embedding_counts = {
            person_id: len(items)
            for person_id, items in grouped.items()
        }
        identity_source_counts = {
            person_id: len({str(item["source_path"]) for item in items})
            for person_id, items in grouped.items()
        }
        for person_id, items in grouped.items():
            gallery_matrix = _normalize(
                np.stack([item["embedding"] for item in items], axis=0)
            )
            similarity = probes @ gallery_matrix.T
            k = min(max(1, int(top_k_per_identity)), similarity.shape[1])
            top_scores = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
            identity_scores[person_id] = top_scores.mean(axis=1)
            identity_names[person_id] = str(items[0]["display_name"])

        sequence_scores = {
            person_id: float(scores.mean())
            for person_id, scores in identity_scores.items()
        }
        ranked = sorted(sequence_scores.items(), key=lambda item: item[1], reverse=True)
        best_person, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else None
        margin = best_score - second_score if second_score is not None else None

        if threshold is None:
            state = "closed_set"
            accepted = True
        elif best_score < float(threshold):
            state = "unknown"
            accepted = False
        elif margin is not None and margin < float(min_margin):
            state = "low_confidence"
            accepted = False
        else:
            state = "stable"
            accepted = True

        window_results = []
        for index in range(probes.shape[0]):
            per_window = sorted(
                (
                    (person_id, float(scores[index]))
                    for person_id, scores in identity_scores.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            window_results.append(
                {
                    **batch.windows[index],
                    "person_id": per_window[0][0],
                    "display_name": identity_names[per_window[0][0]],
                    "similarity": per_window[0][1],
                }
            )

        candidates = [
            {
                "person_id": person_id,
                "display_name": identity_names[person_id],
                "similarity": score,
                "gallery_sources": identity_source_counts[person_id],
                "gallery_embeddings": identity_embedding_counts[person_id],
            }
            for person_id, score in ranked[:5]
        ]
        return {
            "operation": "recognize",
            "state": state,
            "accepted": accepted,
            "candidate_person_id": best_person,
            "candidate_display_name": identity_names[best_person],
            "person_id": best_person if accepted else None,
            "display_name": identity_names[best_person] if accepted else "Unknown",
            "similarity": best_score,
            "second_best_similarity": second_score,
            "similarity_margin": margin,
            "threshold": threshold,
            "min_margin": min_margin,
            "model_key": model_key,
            "source": source,
            "source_fingerprint": source_fingerprint,
            "num_probe_clips": int(probes.shape[0]),
            "num_gallery_embeddings": len(gallery),
            "num_gallery_identities": len(grouped),
            "top_k_per_identity": int(top_k_per_identity),
            "matching_strategy": "mean_probe_of_top_k_gallery_per_identity",
            "gallery_embeddings_per_identity": identity_embedding_counts,
            "gallery_sources_per_identity": identity_source_counts,
            "top_candidates": candidates,
            "window_results": window_results,
            "source_metadata": batch.source_metadata,
            "warning": (
                "No Unknown threshold was supplied; this is closed-set recognition."
                if threshold is None
                else None
            ),
        }
