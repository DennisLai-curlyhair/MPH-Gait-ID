# Live Performance Benchmark Guide

[Documentation index](README.md) | [Traditional Chinese](PERFORMANCE_BENCHMARK_GUIDE_zh.md)

## Purpose

The fourth application page measures the live processing pipeline:

```text
Kinect capture -> person detection -> foreground preprocessing
               -> model encoding -> gallery matching
```

Sensor data are processed in memory. Reports contain timing, FPS, counts,
model configuration, and device information; they do not store RGB or depth
images, point-cloud arrays, or embeddings. Previews are disabled during this
measurement, so the results do not include the normal live-preview rendering
workload.

## Standard Settings

| Setting | Suggested value |
|---|---|
| Detector | YOLO person segmentation |
| Compute device | `auto`; confirm the actual backend uses CUDA for a GPU comparison |
| Clip length | Bundle default; the included bundles use 15 frames |
| Inference stride | 5 |
| Warm-up | 10 seconds |
| Measurement | 60 seconds |
| Subject state | Stationary, for efficiency only |
| Repetitions | At least three per model |

Keep every condition except the model fixed. For example, using YOLO-Seg for
MPH-Gait but SAM for LidarGait++ does not isolate model cost. MPH-Gait ID is the
application; MPH-Gait, PointNet-TMax, and LidarGait++ are the compared encoders.

## Before Measurement

1. Close Teams, Webex, or other applications that may hold the Kinect device.
2. Stop live identification or enrollment on the Real-time page. That page and
   the benchmark page cannot use Kinect simultaneously.
3. Keep only one subject in the detection region.
4. Fix camera position, subject distance, chair, background, lighting, and pose.
5. To compare gallery costs, use comparable numbers of identities and
   descriptors for each model, with galleries enrolled under compatible weights.
6. Run a short, approximately 20-second PointNet-TMax trial first. Confirm
   valid foreground point counts rather than a persistent waiting-person state.

From the repository root, check camera access with:

```bash
python mph_gait_id/scripts/doctor.py --realtime --kinect
```

This attempts to open the camera; USB enumeration alone is not sufficient.

## Measurement Procedure

1. Open **Live Performance Benchmark** and select the model.
2. Set the detector, compute device, stride, warm-up, and measurement duration.
3. Select stationary efficiency measurement. **Expected Person ID** can remain
   empty for this mode.
4. Start the benchmark. Model loading and camera opening are outside the
   formal measurement interval.
5. Keep the subject visible. Warm-up begins only after the first valid human
   point cloud.
6. During measurement, do not move the camera, change settings, or open other
   camera applications or computational workloads.
7. Let the measurement finish; it stops automatically and saves a JSON report.
8. Repeat for each model under the same conditions, at least three times.
   Alternate model order between repetitions.

Example two-model order:

```text
Round 1: MPH-Gait -> LidarGait++
Round 2: LidarGait++ -> MPH-Gait
Round 3: MPH-Gait -> LidarGait++
```

Alternating order reduces systematic warm-up and environmental timing effects.
Include PointNet-TMax in the same controlled schedule when benchmarking all
three encoders.

## Reading the Metrics

| Metric | Meaning |
|---|---|
| `E2E FPS` | Frames processed per second over the measured pipeline interval. |
| `Valid FPS` | Frames yielding valid foreground human points per second. |
| `Model P50` | Median model-encoding latency. |
| `Model P95` | 95th-percentile model-encoding latency; the slowest approximately 5% of measurements lie above it. |
| `Gallery P50` | Median gallery-operation latency, including gallery access and matching in the measured path. |
| `Valid %` | Fraction of processed frames yielding valid human point clouds. |
| `Recognition updates/s` | Inference-result updates per second, affected by valid-frame throughput and inference stride. |
| `gpu_memory.peak_allocated_mib` | Peak PyTorch GPU allocation during measurement; unavailable for CPU-only measurement. |

Low `Valid %` can make encoder timing comparisons unrepresentative because
detection or foreground extraction is failing. Examine both throughput and
valid-frame coverage, not just instantaneous FPS. Model and gallery latencies
are measured on inference events, not on every captured frame.

Compare repeated-run trends rather than selecting the highest FPS. Investigate
occlusion, extra people, or insufficient foreground points when coverage drops.
Record any exclusion and its reason instead of silently removing slow runs.

## Stationary-Test Limits

A stationary subject stabilizes position, foreground density, and background;
the encoder still receives fixed-size `T x N` input sequences. This is useful
for efficiency comparison but cannot establish:

- Gait identification accuracy.
- Appropriate unknown-rejection thresholds.
- Time to stable identification during walking.
- Robustness to walking occlusion or changed viewing direction.

To inspect identification behavior, use walking mode, provide the expected
person ID, and repeat a fixed route with multiple subjects. A controlled
accuracy or open-set evaluation still requires its own protocol.

## Reports and Export

Runs containing formal measurement data are saved automatically under:

```text
outputs/performance_benchmarks/benchmark_*.json
```

Select a result row and export the selected report:

- **JSON:** retains the full hierarchy and interpretation notes.
- **CSV:** flattens the report into one row for spreadsheet or statistical tools.

Stopping early after measurement has begun saves partial results marked as
incomplete. Stopping while waiting for a person or before formal measurement
does not create an empty report.
