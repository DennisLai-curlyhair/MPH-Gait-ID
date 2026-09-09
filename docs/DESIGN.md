# Real-Time Point-Cloud Gait Identification System Design

[Documentation index](README.md) | [Traditional Chinese](DESIGN_zh.md)

## Scope

The gait encoders consume foreground human point clouds, not full-scene point
clouds or RGB images. Supported architectures are PointNet-TMax (`pc_v1` in
internal configuration), MPH-Gait (`mph_gait`), and adapted LidarGait++
(`lidargaitpp`). Each encoder receives a fixed-length window of XYZ point sets.

The real-time pipeline consists of device capture, person segmentation,
point-cloud preprocessing, window sampling, model inference, and gallery
matching. RGB is used for person detection and visual feedback, not as input
to the gait encoder.

```text
Azure Kinect RGB / depth / organized XYZ
  -> primary-person detection or segmentation
  -> calibrated foreground-point extraction and filtering
  -> point sampling and fixed-length window
  -> selected gait encoder
  -> compatible gallery matching and identity decision
```

## Gallery Compatibility

Descriptors are isolated by their model and input-processing configuration.
Compatibility covers the model bundle and checkpoint, coordinate adapter,
frame-window length, initial-frame discard setting, and preprocessing profile.
Changing these settings requires a compatible gallery; descriptors from
different models must not be compared as if they occupied the same space.

The UI calls the preprocessing profile the **point-cloud processing version**.
The database retains the field name `preprocessing_profile_id` for schema
compatibility. This identifies the foreground filtering and normalization
rules used to produce the model input.

See [Final-24 deployment weights](FINAL24_WEIGHTS.md) before switching from
Fixed-Special5 to Final-24 bundles. The existing gallery is retained, but
identities must be enrolled under the new compatible bundle.

## Foreground Extraction

YOLO detection selects a primary person using a bounding box. YOLO segmentation
or YOLO-prompted SAM can supply a tighter person mask. Kinect calibration aligns
the image support with XYZ observations; depth-range filtering and point-cloud
cleaning then remove invalid or unwanted points.

Enrollment allows multiple visible people by default but collects only the
selected primary person. The UI can instead pause collection whenever multiple
people are detected. Identification also uses the primary-person policy. These
policies do not provide multi-person tracking or guarantee identity continuity
when people cross or occlude one another.

The final depth-filtered mask can be mapped back to RGB as a green overlay.
This visualization does not change the model window, extracted descriptors, or
gallery writes. It is not a separate image-based gait model.

## Timing and Windows

Device capture and replay use monotonic timestamps. Window metadata records
the actual first and last frames, effective sampling FPS, mean frame gap, and
maximum gap. Backward timestamps, zero gaps, or gaps above the configured limit
clear the sequence buffer rather than joining discontinuous frames into one
walking window.

## Guided Enrollment

Each walking pass records its capture type, observed movement direction,
displacement, and motion monotonicity independently. The current models were
trained on front-view sequences, so capture offers only the front-facing type.
The operator must keep turns and backward return walks outside recorded passes.

Trajectory diagnostics measure translation, not whether the body faces toward
or away from the camera. They provide warnings rather than an unconditional
rejection: the operator can include a non-discarded pass with embeddings after
reviewing its quality warning.

Movement diagnostics use the median foreground X/Z center of each frame in
camera coordinates. Start and end positions use medians over roughly the first
and last 20% of the trajectory, with 2-10 frames per segment. A depth change of
approximately 250 mm distinguishes approaching from departing; a lateral
change of approximately 150 mm indicates sideways movement. At least 65% of
adjacent steps along the dominant axis must agree in direction, allowing
10 mm jitter. Otherwise the pass is flagged as stationary or possibly turning.
Front-facing capture expects approaching movement, but the operator must
confirm actual orientation.

Starting a session or ending a pass does not commit enrollment. The selected
passes are written only after review and confirmation. Person, model, and
source records are committed in one SQLite transaction. A failure in any source
rolls back the batch to avoid partially registered identities.

## Gallery Management

Gallery Manager filters records by the current model bundle, frame length, and
point-cloud processing version. Individual sources or passes can be disabled
and restored. All descriptors for one person within the current compatibility
scope can also be disabled without affecting another model's gallery.

## Live Performance Measurement

The benchmark page uses the same Kinect, detector, foreground preprocessing,
encoder, and gallery pipeline without generating image previews. The pipeline
thread sends scalar telemetry to a dedicated recorder before the UI queue, so
slow Tk updates or dropped preview snapshots do not discard measurement frames.

Model encoding and gallery matching are timed separately. Warm-up starts after
the first valid foreground point cloud, followed by a timed measurement period.
The recorder retains numerical samples, state counts, and run configuration;
it does not retain or serialize RGB, depth, point-cloud arrays, or embeddings.
Stationary reports are marked `stationary_efficiency_only` and are not evidence
of identification accuracy. See the [benchmark guide](PERFORMANCE_BENCHMARK_GUIDE.md).

## Validation Status

- Unknown-rejection threshold calibration: **TBD**
- Azure Kinect hardware FPS: **TBD**
- Long-duration stability: **TBD**
- Person-mask alignment: **TBD**
