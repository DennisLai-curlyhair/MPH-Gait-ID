from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WorkerMessage:
    kind: str
    payload: Any


class BackgroundWorker:
    """Run one model operation at a time without blocking Tk's event loop."""

    def __init__(self) -> None:
        self.messages: queue.Queue[WorkerMessage] = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task: Callable[[Callable[[str], None]], Any]) -> None:
        if self.busy:
            raise RuntimeError("Another operation is already running")
        self._thread = threading.Thread(
            target=self._run,
            args=(task,),
            name="gait-system-worker",
            daemon=True,
        )
        self._thread.start()

    def publish_progress(self, message: str) -> None:
        self.messages.put(WorkerMessage("progress", str(message)))

    def drain(self) -> list[WorkerMessage]:
        values: list[WorkerMessage] = []
        while True:
            try:
                values.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return values

    def _run(self, task: Callable[[Callable[[str], None]], Any]) -> None:
        try:
            result = task(self.publish_progress)
            self.messages.put(WorkerMessage("result", result))
        except Exception as exc:
            self.messages.put(
                WorkerMessage(
                    "error",
                    {
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
