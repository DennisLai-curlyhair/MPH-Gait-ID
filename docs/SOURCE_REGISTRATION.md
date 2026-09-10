# Multi-Model Source Registration

Stage C re-encodes the opt-in foreground recordings saved by
[Stage B](ENROLLMENT_SOURCES.md). One job can register selected passes to
several installed model/checkpoint bundles. It does not train or fine-tune models,
convert an existing embedding into another model's embedding, or require a camera.

## Desktop Workflow

1. Stop live recognition, enrollment, and benchmarking; commit or abandon any
   pending enrollment review. Back up the Gallery before testing an upgrade.
2. Open **Enrollment sources**, select one source recording, and inspect its
   person, passes, and playback.
3. Select **Register to models**. Check the passes to reuse and each target
   model/checkpoint. For example, select PointNet-TMax, MPH-Gait, and LidarGait++.
4. Set **T** independently for each target. It defaults to the bundle's clip
   length. **N** comes from the bundle and is not overridden here. Set the maximum
   embeddings per pass (default 10) and processing device (default `auto`).
5. Select **Register selected models** and confirm. Models load sequentially;
   progress shows the current target and processed windows. Controls remain
   responsive. **Cancel job** waits for the current inference call to return.
6. Review the per-model/per-pass result. Switch to Gallery or live recognition
   and select the same target bundle and T to use the newly registered person.

One source recording may contain multiple reviewed walking passes. This release
processes one source recording per job, with multiple passes and multiple targets.
Separate source recordings are submitted as separate jobs. Only selected,
installed point-cloud bundles are used; no weights are downloaded automatically.

## Encoding Contract

```text
Saved foreground XYZ (variable N, camera coordinates, mm)
  -> verify source manifest and used frame hashes
  -> resolve source owner's persistent UID
  -> split each selected pass at sampling interruptions
  -> form non-overlapping complete T-frame windows
  -> sample each frame to the target's N points
  -> existing ModelAdapter (axis mapping, model-specific normalization)
  -> target checkpoint forward pass
  -> finite, nonzero, L2-normalized descriptor
  -> atomically add all new target galleries
```

Sampling uses the same deterministic point sampler as live foreground processing.
The model adapter performs the existing camera-to-model axis conversion, and
LidarGait++ retains its normalized XYZ plus metric height input. Embeddings are
separated by the existing model/checkpoint/T/preprocessing compatibility keys.
The source's preprocessing profile must match the application's profile; stored
foreground cannot be relabeled as if a newer background-removal pipeline had run.
Source recording files, weights, model implementations, and live encoding are
not rewritten by Stage C.

These are already accepted enrollment frames: **the first 30 frames are not
discarded again**. The bundle's `drop_first_frames` value remains in the existing
model compatibility record for consistency with live enrollment, but this field
does not trigger an extra trim of saved foreground recordings.

Stride equals T. Incomplete tails are discarded, not padded or repeated.
Explicit segment breaks, non-increasing timestamps/frame indices, and gaps
exceeding the saved maximum valid gap break a window. Windows never cross pass
boundaries. If more windows are available than the per-pass limit, complete
windows are selected evenly across that pass's eligible windows. For example,
120 uninterrupted saved frames provide eight windows at T=15 or four at T=30.
The per-pass limit applies separately to each model.

## Duplicates and Failures

- **already_registered:** that source/pass already has Gallery rows under the
  target model key, including the original Stage B enrollment. Both active and
  inactive rows count. No additional embeddings are created for that pass.
- **too_short:** no uninterrupted complete window exists at the selected T.
  Other eligible passes/targets may still be registered.
- **registered:** all new descriptors were committed successfully.
- **skipped:** no new eligible pass remains for this target; its model is not loaded.
- **failed/cancelled:** the job adds no embeddings for any target. Existing
  galleries and recordings are unchanged. Retry after correcting the problem.

Inactive passes are never silently reactivated. Reactivate them in Gallery
management. If all embeddings of a pass were permanently deleted, explicitly
confirming a new registration can recreate descriptors from the retained source
as new records. This is re-encoding, not tombstone restoration or archive import.
If only some embeddings of a pass remain, the entire pass is conservatively
skipped; Stage C is not an automatic per-window repair tool.

Deleted/inactive source owners are blocked. Restore the original person through
Gallery management or an authorized Stage A archive restore first. Reusing the
same visible Person ID for another person does not authorize access to the old
recording, and Stage C does not automatically reassign orphan sources.

All eligible targets are computed before a single SQLite transaction publishes
their descriptors and completion record. A failure in a later target or in the
database transaction rolls back this job's earlier additions. Owner, target
bundle/checkpoint, and duplicate checks run again before commit. A source-library
OS lock blocks concurrent source deletion, cleanup, or capture by another app
instance. The desktop also blocks competing Gallery/camera/offline jobs.
Other external GPU applications are not controlled by this lock.

Completed, failed, and cancelled jobs are recorded in the local
`enrollment_source_jobs` SQLite table, including source ID/hash, owner UID,
target hashes/T/N, pass status, and added counts. Failure logging is best-effort
if the database itself is unavailable. An abrupt process termination can omit a
failure log; SQLite still prevents a partially committed multi-model Gallery.
Stage A `.mphgallery` exports contain descriptors, not foreground media or this
local job history. Protect the database and recordings as biometric data.

## Verification

From the repository root:

```bash
python -m unittest discover -s mph_gait_id/tests
python scripts/smoke_source_registration.py
```

The smoke test uses temporary synthetic foreground frames and the three installed
Final-24 bundles on CPU. It compares Stage C output with the same live-window
encoding path, checks model keys and archive import, and leaves no real Gallery
entries. It does not access Kinect or download weights. To test one bundle:

```bash
python scripts/smoke_source_registration.py --bundle mph_gait_final24_len15_seed2
```

Local acceptance checks:

- Register one Stage B recording to two or three different checkpoints.
- Verify the person appears under each selected bundle and T.
- Repeat the job: previously registered passes must be skipped.
- Cancel a larger job: no target's Gallery count may increase.
- Test short/discontinuous passes and per-target T=15/T=30.
- Verify source playback/deletion, Gallery deactivation/deletion, and Stage A
  export/import still work. Test both UI languages and desktop scaling.

Source recordings cannot recover background or RGB/depth pixels that were not
saved. Re-running segmentation or changing the foreground extraction algorithm
requires new sensor data. Open-set thresholds remain model-specific and require
separate calibration; re-encoding does not transfer calibrated thresholds.
