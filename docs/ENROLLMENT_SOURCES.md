# Enrollment Foreground Source Library

Stage B preserves enrollment foreground point clouds for inspection and future
re-encoding. It does not implement cross-model registration. Stage A Gallery
transfer remains descriptor-only: `.mphgallery` archives do not include these
recordings.

## Capture and Review

1. Open **Real-time Azure Kinect**, choose **Enrollment**, and enter the identity.
2. Enable **Consent to save enrollment foreground points** before starting the
   session. This option is off at every application launch. Obtain the subject's
   permission and apply the deployment's data-retention policy.
3. Start the session and record passes as usual. Device discovery, segmentation,
   foreground filtering, sampling, and identity encoding remain unchanged.
4. Select the passes in **Finish & review** and commit. Only selected passes are
   published in **Enrollment sources**. A manually accepted movement-warning
   pass retains the quality warning and override record.
5. Open **Enrollment sources**, select a recording and a pass, and use Play,
   Pause, Previous/Next, the frame slider, and zoom. Playback is a front-view
   rendering of stored XYZ, not a new inference run or an RGB video.

The recording includes valid frames from selected passes, even when Gallery
registration retains only a subset of their embeddings. It excludes countdown
frames, no-person frames, discarded passes, and unselected passes. Recognition
does not record sources, regardless of the enrollment checkbox.

Review can be closed and reopened or resumed without losing temporary points.
Discard pass deletes that pass's temporary points. Abandon, normal capture
cancellation, and application exit remove uncommitted temporary frames. Closing
the application during a Gallery commit is blocked until the operation finishes.

## Data Boundary

```text
RGB + organized XYZ
    -> person selection and foreground filtering
        -> optional background source writer: variable N x 3 FP32, camera XYZ mm
        -> existing fixed-point sampler -> runtime axis/normalization -> encoder

Review-selected passes
    -> verified source files + manifest
    -> one SQLite transaction: identity / embeddings / source reference
    -> committed source library
```

Stored coordinates retain the camera convention (X right, Y down, Z forward),
millimeters, body scale, and translation. They have not undergone fixed-point
sampling, centering, axis reordering, or scale normalization. Source data are
model-independent; embeddings are still specific to their model, checkpoint,
preprocessing, point count, and window length.

No RGB images, instance-mask images, raw depth maps, or full-scene organized
point clouds are saved by Stage B. Foreground extraction can still contain
segmentation errors. Removing background at capture is irreversible: this library
cannot rerun segmentation or reconstruct discarded sensor observations.

## Files and Provenance

The source root follows the configured database path. For the default
`data/gallery.sqlite3` it is:

```text
data/
  gallery.sqlite3
  gallery.sqlite3_enrollment_sources/
    .writer.lock
    staging/<source-uuid>/       # temporary frame files and frames.jsonl
    records/<source-uuid>/
      manifest.json
      frame_000000.npy
      ...
    trash/                      # deletion recovery until completion
```

The `enrollment_sources` SQLite table stores immutable source UUIDs, owner
transfer UIDs, captured names/IDs, counts, byte sizes, and manifest SHA256. No
model or existing Gallery schema migration is required. Frame NPY files contain
only numeric FP32 arrays and are loaded with `allow_pickle=False`.

The versioned `mph-foreground-source-v1` manifest records:

- Frame filenames, point counts, SHA256, sensor frame indices, timestamps, and
  segment boundaries. Timestamps use the source clock, not assumed wall time.
  Session creation time is UTC. Do not join frames across passes or segment breaks.
- Requested/observed direction, quality summaries, manual review decisions,
  selected windows, and original local embedding IDs.
- Coordinate convention, unit, storage stage, and opt-in consent flag.
- Sensor metadata supplied by the capture adapter; SDK color alignment or replay
  projection metadata; detector settings, filter parameters, preprocessing source
  hash, and the model/checkpoint record used at original enrollment.

The Azure adapter performs calibrated SDK alignment upstream; Stage B does not
claim to archive a full sensor calibration object. Replay projection parameters
are retained when the adapter supplies them. Hashes detect modification/corruption;
they are not encryption or a signature against a malicious database owner.

## Storage and Failure Handling

Defaults in `mph_gait_id/configs/system.yaml`, under `realtime.foreground_storage`:

| Setting | Default |
|---|---:|
| `session_limit_bytes` | 2 GiB |
| `library_limit_bytes` | 20 GiB |
| `min_free_bytes` | 512 MiB |
| `queue_size` | 8 frames |
| `max_frame_points` | 2,000,000 |
| `max_frames` | 9,000 per session |

The background writer copies accepted arrays before the native camera buffers
can be reused. A full queue, exceeded quota, or write failure invalidates
source-backed enrollment instead of silently dropping frames. Improve local disk
throughput or restart enrollment with saving disabled; no existing Gallery is
deleted. Persisting foreground points adds disk I/O and is not a runtime-speed
claim.

Source files are prepared before Gallery commit, and the source reference is
inserted in the same SQLite transaction as the embeddings. A failed transaction
removes unpublished source files and keeps staging available for review/retry.
Capture errors cancel temporary recording. A reporting/cleanup warning after a
successful commit is not a request to register the same pass again.

An OS file lock prevents simultaneous source-library writers, deletion during
capture/review, and cleanup by another running app. After an abnormal process
exit, use **Clean uncommitted files** once no session is active. This removes
orphan staging and unpublished files and recovers an interrupted source deletion
whose SQLite row still exists. Committed rows with missing/corrupted files are
reported as errors rather than converted into valid recordings.

## Deletion, Transfer, and Privacy

- **Delete source** permanently removes the selected recording, not Gallery
  embeddings. There is no source recycle bin or embedding-to-point reconstruction.
- Gallery person/fragment deletion leaves sources intact. Delete them separately
  when consent is withdrawn. Local Gallery backups, session JSON, and external
  backups have separate retention policies.
- Renaming a person updates the current-name display; the manifest retains the
  name at capture. `*` after an ID means no current Gallery person is linked.
  Reusing the same visible Person ID does not attach old recordings to a new person.
- Portable Gallery export/import and its backups do not copy source NPY files.
  Source transfer and multi-model re-encoding are separate future stages. For an
  administrative backup, stop the app and back up both the database and its source
  root; copying only the SQLite database is insufficient for source playback.
- Source paths, recordings, temporary files, and runtime databases are ignored by
  Git. No participant data or existing registrations are included in the project.
  Use trusted local storage, OS permissions, encrypted disks/backups where needed,
  and an explicit retention policy. The app does not encrypt source recordings.

## Local Acceptance Tests

```bash
python -m unittest discover -s mph_gait_id/tests
python -m unittest discover -s scripts/tests
python scripts/validate_release.py
python run.py ui
```

With Kinect hardware, verify: opt-out enrollment; opt-in multi-pass capture;
discard/resume/selected-pass commit; source playback and zoom; deletion without
embedding changes; restart persistence; and a full-disk/slow-disk cancellation.
Also verify both UI languages and your desktop scaling. Synthetic tests do not
replace Windows/NTFS, Azure SDK, real-camera, and local GUI acceptance testing.
