"""Presentation helpers; recognition decisions remain owned by the pipeline."""
from __future__ import annotations

from tkinter import ttk


class ControlLock:
    """Temporarily disable controls without losing their original states."""

    def __init__(self, controls):
        self.controls = list(controls)
        self.states = None

    def set_locked(self, locked):
        if locked and self.states is None:
            self.states = [(widget, str(widget.cget("state"))) for widget in self.controls]
            for widget in self.controls:
                widget.configure(state="disabled")
        elif not locked and self.states is not None:
            for widget, state in self.states:
                widget.configure(state=state)
            self.states = None


def form_controls(parent, exclude=()):
    controls = []
    for child in parent.winfo_children():
        if child in exclude:
            continue
        if isinstance(child, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Spinbox,
                              ttk.Checkbutton, ttk.Radiobutton)):
            controls.append(child)
        controls.extend(form_controls(child, exclude))
    return controls


def snapshot_feedback(snapshot, tr):
    """Sensor state takes precedence over an earlier recognition result."""
    state = snapshot.state
    result = snapshot.result or {}
    details = dict(frames=snapshot.buffer_size, total=snapshot.clip_len)
    if state in {"starting", "source_ready", "waiting_person", "multiple_people",
                 "insufficient_points", "collecting", "complete", "error"}:
        return (tr("workflow." + state),
                snapshot.message if state == "error" else tr("workflow." + state + ".detail", **details))
    if snapshot.operation == "enroll":
        state = str(result.get("state", state))
        if state == "enrollment_failed":
            return tr("workflow.enrollment_failed"), str(result.get("error", snapshot.message))
        if state in {"enrollment_paused", "enrollment_countdown", "enrolling", "enrollment_review", "enrolled"}:
            return tr("workflow." + state, name=result.get("display_name", "")), tr(
                "workflow." + state + ".detail",
                seconds=float(result.get("warmup_remaining_s", 0)),
                count=result.get("captured_embeddings", 0),
                minimum=result.get("min_embeddings", 0),
                passes=result.get("usable_passes", 0),
                stored=result.get("stored_embeddings", 0))
    if state == "no_gallery" or result.get("state") == "no_gallery":
        return tr("workflow.no_gallery"), tr("workflow.no_gallery.detail")
    if result:
        candidate = result.get("candidate_display_name") or result.get("display_name", "")
        title = (str(result.get("display_name", "")) if result.get("accepted") else
                 tr("workflow.pending" if result.get("state") == "accumulating" else "workflow.unknown",
                    name=candidate))
        return title, tr("workflow.scores", score=float(result.get("similarity", 0)),
                         margin=float(result.get("similarity_margin") or 0),
                         count=result.get("stability_count", 0), total=result.get("stability_required", 0))
    return tr("workflow.collecting"), tr("workflow.collecting.detail", **details)
