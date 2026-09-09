# MPH-Gait ID User Interface Guide

[Documentation index](README.md) | [Traditional Chinese](UI_GUIDE_zh.md)

Install dependencies and launch the application as described in the
[project README](../README.md). Use the language selector to switch between
English and Traditional Chinese. The control names below refer to English.

## Application Pages

| Page | Purpose |
|---|---|
| Offline registration / recognition | Enroll or identify from existing `clear_data_*.npy` point-cloud sequence folders. |
| Real-time | Connect Azure Kinect DK, preview foreground points, enroll walking passes, and identify people. |
| Gallery Manager | Inspect and manage registered people, sessions, passes, and sources. |
| Live Performance Benchmark | Measure pipeline and model efficiency using Kinect; save numerical reports without previews. |

**Open Offline batch** continues the offline workflow. Gallery maintenance is
handled in Gallery Manager rather than mixed into live capture controls.
For moving registered features to another computer, use **Export all Gallery**
and **Import Gallery** on that page. See [Gallery transfer](GALLERY_TRANSFER.md)
for compatibility, identity conflicts, and backup behavior.

## Live Preview

The real-time page has two previews: RGB person detection and foreground human
point clouds. The cloud widget keeps a stable size as point count or subject
size changes; use the mouse to zoom the view.

The RGB point-cloud overlay toggle displays foreground positions in green
after depth and foreground filtering. It affects only the preview, not the gait
model input. There is no separate image-based gait input in this application.

## Person Extraction Modes

- **YOLO:** uses a person bounding box; nearby background inside the box may
  remain in the extracted points.
- **YOLO-Seg:** uses an instance mask and is the default real-time option.
- **SAM:** refines the person region using a YOLO bounding-box prompt. It does
  not independently search for all people and requires optional dependencies
  and weights. Its additional processing can reduce throughput.

## Guided Enrollment

Use front-facing walking passes, consistent with the training view. Begin
recording before the forward walk and stop at its end. Do not include turning
or the return walk in the same pass.

1. Select the model bundle and compatible frame length. Enter a stable person
   ID and, optionally, a display name.
2. Select enrollment and start the guided enrollment session. Wait until the
   session is ready; this alone does not record a walking pass.
3. Press **Start pass** before the person begins the front-facing walk.
4. Press **End pass** after the walk, or **Discard pass** to reject it.
5. Record further passes as needed, then press **Finish & review**.
6. Select the passes to include. A pass must contain embeddings and must not
   have been discarded. Movement or quality warnings require operator review;
   they do not automatically prevent manual inclusion.
7. Use **Continue adding passes** to return to capture, or **Commit selected
   passes** to write the selection to the gallery.
8. Confirm the success message and the number of saved clips, then check the
   identity in Gallery Manager.

Closing the review window is not a commit. The review can be reopened within
the same application session. **Abandon session** rejects the pending session;
it is not required merely to continue collecting passes.

Allowing enrollment when multiple people are visible is enabled by default.
Only the selected primary person is collected. Disable this option to pause
collection on multiple-person detections, particularly if another person may
cross in front of the intended subject. This is not multi-person tracking.

Automatic direction diagnostics estimate translation, not body orientation.
The operator must verify that selected passes actually show the intended
front-facing person.

## Live Identification

Select a model with compatible registered gallery entries and start live
identification. Complete rolling windows are encoded and matched to that
gallery. The UI shows the candidate identity, similarity, and processing rate.
Unknown rejection uses the configured score, score-margin, and temporal
stability settings; their defaults are not a validated open-set calibration.

When changing model weights, verify gallery compatibility rather than reusing
descriptors from a different checkpoint. See the
[Final-24 weight guide](FINAL24_WEIGHTS.md) for re-enrollment requirements.

## Live Performance Benchmark

The fourth page disables image previews to avoid rendering overhead in the
pipeline comparison. After selecting a model and starting the benchmark, the
first valid human point cloud triggers warm-up. Formal measurement follows
automatically and produces a numerical JSON report.

A stationary subject can support efficiency measurements for capture,
detection, preprocessing, model encoding, and gallery matching. This does not
measure walking identification accuracy. Keep detector, compute device,
inference stride, subject position, and gallery size consistent across model
comparisons. See the [benchmark guide](PERFORMANCE_BENCHMARK_GUIDE.md).

## Point-Cloud Processing Version

This setting identifies the foreground filtering and normalization rules used
to produce the input points. Changed depth ranges or sampling rules may require
a different processing version. Features from incompatible versions should
not be mixed in retrieval; Gallery Manager exposes this filter. For ordinary
operation, retain the default human-foreground point-cloud version.
