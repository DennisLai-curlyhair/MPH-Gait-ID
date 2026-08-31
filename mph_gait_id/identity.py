from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PERSON_PATTERN = re.compile(r"(?:^|[_-])P(?:ERSON)?[_-]?(\d+)", re.IGNORECASE)
CLOTHES_PATTERN = re.compile(
    r"(?:^|[_-])C(?:LOTH(?:ES)?)?[_-]?(\d+)",
    re.IGNORECASE,
)
VIDEO_PATTERN = re.compile(r"(?:^|[_-])V(?:IDEO)?[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RegistrationIdentitySuggestion:
    person_id: str
    default_display_name: str
    clothes_ids: tuple[int, ...]
    video_ids: tuple[int, ...]
    source_count: int

    @property
    def detail(self) -> str:
        clothes = ", ".join(f"C{value}" for value in self.clothes_ids) or "-"
        captures = ", ".join(f"V{value:03d}" for value in self.video_ids) or "-"
        return (
            f"偵測到 {self.person_id} | 服裝 {clothes} | "
            f"拍攝編號 {captures} | {self.source_count} 個來源"
        )


def _candidate_names(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    first = resolved.stem if resolved.is_file() else resolved.name
    names = [first]
    names.extend(parent.name for parent in list(resolved.parents)[:3])
    return [name for name in names if name]


def _first_match(pattern: re.Pattern[str], names: Iterable[str]) -> int | None:
    for name in names:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def parse_sequence_metadata(path: str | Path) -> dict[str, object]:
    source = Path(path)
    names = _candidate_names(source)
    person_id = _first_match(PERSON_PATTERN, names)
    clothes_id = _first_match(CLOTHES_PATTERN, names)
    video_id = _first_match(VIDEO_PATTERN, names)
    return {
        "known_dataset_sequence": all(
            value is not None for value in (person_id, clothes_id, video_id)
        ),
        "person_id": person_id,
        "person_key": None if person_id is None else f"P{person_id:03d}",
        "clothes_id": clothes_id,
        "video_id": video_id,
    }


def suggest_registration_identity(
    sources: Iterable[str | Path],
) -> RegistrationIdentitySuggestion | None:
    metadata = [parse_sequence_metadata(source) for source in sources]
    person_ids = {
        int(item["person_id"])
        for item in metadata
        if item.get("person_id") is not None
    }
    if not person_ids:
        return None
    if len(person_ids) != 1:
        detected = ", ".join(f"P{value:03d}" for value in sorted(person_ids))
        raise ValueError(
            "註冊來源包含不同人物，請分開註冊。"
            f"目前偵測到：{detected}"
        )
    person_number = next(iter(person_ids))
    person_key = f"P{person_number:03d}"
    clothes_ids = tuple(
        sorted(
            {
                int(item["clothes_id"])
                for item in metadata
                if item.get("clothes_id") is not None
            }
        )
    )
    video_ids = tuple(
        sorted(
            {
                int(item["video_id"])
                for item in metadata
                if item.get("video_id") is not None
            }
        )
    )
    return RegistrationIdentitySuggestion(
        person_id=person_key,
        default_display_name=person_key,
        clothes_ids=clothes_ids,
        video_ids=video_ids,
        source_count=len(metadata),
    )
