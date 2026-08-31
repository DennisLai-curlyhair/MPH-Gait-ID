#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.controller import GaitApplicationController  # noqa: E402
from mph_gait_id.realtime.pipeline import (  # noqa: E402
    RealtimeConfig,
    RealtimePipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pass -> review -> resume -> pass -> review -> commit "
            "without replacing the enrollment session."
        )
    )
    parser.add_argument(
        "--replay",
        default=str(SYSTEM_ROOT / "data" / "replay" / "P3_C4_V001"),
    )
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--embeddings-per-pass", type=int, default=2)
    return parser


def capture_round(
    pipeline: RealtimePipeline,
    *,
    direction: str,
    target_total: int,
    deadline: float,
    states: Counter[str],
) -> dict[str, object]:
    pipeline.start_enrollment_pass(direction)
    pass_ended = False
    finish_requested = False
    while time.perf_counter() < deadline:
        snapshot = pipeline.poll_latest()
        if snapshot is None:
            time.sleep(0.02)
            continue
        states[snapshot.state] += 1
        if snapshot.state == "error":
            raise RuntimeError(snapshot.message)
        if snapshot.enrollment_embeddings >= target_total and not pass_ended:
            pipeline.end_enrollment_pass(False)
            pass_ended = True
        passes = pipeline.enrollment_passes()
        if (
            pass_ended
            and passes
            and passes[-1].get("ended_at") is not None
            and not finish_requested
        ):
            pipeline.finish_enrollment_for_review()
            finish_requested = True
        if snapshot.state == "enrollment_review":
            return dict(snapshot.result or {})
    raise TimeoutError("Timed out before enrollment review")


def wait_until_stopped(pipeline: RealtimePipeline, timeout: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout
    while pipeline.running and time.perf_counter() < deadline:
        time.sleep(0.02)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay = Path(args.replay).expanduser().resolve()
    target_per_pass = max(1, int(args.embeddings_per_pass))
    states: Counter[str] = Counter()
    started = time.perf_counter()
    deadline = started + float(args.timeout)

    with TemporaryDirectory(prefix="mph-gait-id-review-resume-") as raw:
        root = Path(raw)
        controller = GaitApplicationController(
            database_path=root / "gallery.sqlite3",
            output_root=root / "outputs",
        )
        pipeline = RealtimePipeline(
            controller,
            RealtimeConfig(
                bundle_id="pointnet_tmax_fixed_special5_seed0_split0",
                operation="enroll",
                source_mode="replay",
                replay_path=str(replay),
                replay_fps=float(args.fps),
                replay_loop=False,
                detector="replay_depth",
                device="auto",
                clip_len=15,
                num_points=1024,
                enrollment_person_id="MPH_GAIT_ID_RESUME_SELFTEST",
                enrollment_display_name="MPH-Gait ID Resume Self Test",
                enrollment_note="temporary validation only",
                enrollment_duration_s=float(args.timeout),
                enrollment_warmup_s=0.1,
                enrollment_min_embeddings=2,
                enrollment_max_embeddings=10,
                enrollment_stride=5,
                enrollment_session_root=str(root / "sessions"),
                guided_passes=True,
            ),
        )
        try:
            pipeline.start()
            first_review = capture_round(
                pipeline,
                direction="left_to_right",
                target_total=target_per_pass,
                deadline=deadline,
                states=states,
            )
            wait_until_stopped(pipeline)
            session_id = pipeline._session_id
            first_count = int(first_review.get("captured_embeddings", 0))

            pipeline.resume_enrollment_capture()
            second_review = capture_round(
                pipeline,
                direction="right_to_left",
                target_total=first_count + target_per_pass,
                deadline=deadline,
                states=states,
            )
            wait_until_stopped(pipeline)
            if pipeline._session_id != session_id:
                raise RuntimeError("Enrollment session ID changed after review resume")
            if states["starting"] < 2 or states["source_ready"] < 2:
                raise RuntimeError(
                    "Each capture round must report background startup and source-ready states"
                )

            passes = pipeline.enrollment_passes()
            selected = [
                str(item["pass_id"])
                for item in passes
                if not item.get("discarded")
                and int(item.get("embedding_count", 0)) > 0
            ]
            if len(selected) < 2:
                raise RuntimeError(
                    f"Expected two usable passes after resume, found {len(selected)}"
                )
            result = pipeline.commit_enrollment(selected)
            db_passes = controller.list_gallery_passes(
                "pointnet_tmax_fixed_special5_seed0_split0",
                "MPH_GAIT_ID_RESUME_SELFTEST",
                clip_len=15,
                include_inactive=True,
            )
        except Exception as exc:
            pipeline.stop()
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "states": dict(states),
                        "passes": pipeline.enrollment_passes(),
                    },
                    indent=2,
                )
            )
            return 1

        print(
            json.dumps(
                {
                    "status": "ok",
                    "replay": str(replay),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "session_id_preserved": pipeline._session_id == session_id,
                    "first_review_embeddings": first_count,
                    "second_review_embeddings": int(
                        second_review.get("captured_embeddings", 0)
                    ),
                    "passes": passes,
                    "selected_pass_ids": selected,
                    "stored_embeddings": int(result.get("stored_embeddings", 0)),
                    "database_pass_rows": len(db_passes),
                    "formal_gallery_modified": False,
                    "states": dict(states),
                },
                indent=2,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
