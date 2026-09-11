"""Atomic operation reports, separate from committed Gallery transactions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def write_result(output_dir: Path, result: dict[str, Any]) -> Path:
    target = output_dir / "result.json"
    payload = json.dumps(
        {**result, "result_path": str(target)}, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    fd, name = tempfile.mkstemp(prefix=".result-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    result["result_path"] = str(target)
    return target


def report_warning(result: dict[str, Any], error: Exception) -> None:
    result.pop("result_path", None)
    result["report_warning"] = (
        "Operation completed, but its JSON report could not be saved. "
        "Gallery changes remain committed; do not repeat enrollment. "
        f"{type(error).__name__}: {error}"
    )
