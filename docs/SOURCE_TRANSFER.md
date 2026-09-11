# Portable Enrollment Source Transfer

Stage D moves the saved foreground point-cloud recordings from Stage B between
installations. It complements descriptor-only [Gallery transfer](GALLERY_TRANSFER.md)
and [multi-model source registration](SOURCE_REGISTRATION.md). No new dependency,
camera, or model inference is required for export/import itself.

## What Each Archive Contains

| Content | `.mphsources` | `.mphgallery` |
|---|---|---|
| Saved foreground XYZ arrays | Yes, selected committed sources | No |
| Point counts, timestamps, pass/segment boundaries | Yes | Window metadata only |
| Coordinate convention and foreground preprocessing provenance | Yes | Encoder compatibility metadata |
| Portable identity links and names | Yes | Yes |
| Model-specific embeddings | No | Yes |
| Model weights, raw RGB/depth, full-scene point clouds | No | No |

These are **foreground points**, not the sensor's raw scene or an embedding.
Arrays remain variable-size float32 XYZ in camera coordinates (X right, Y down,
Z forward), in mm, before fixed-point sampling, centering, or axis conversion.
Stored segmentation/filtering cannot be undone or rerun without raw observations.

Only committed recordings are selectable. A recording includes all saved frames
from its review-selected passes, not just windows that produced embeddings.
Export selection is recording-level, not individual frames or passes.

## A-to-B Workflow

1. Finish or abandon pending capture/review and stop other application jobs.
2. On A, open **Enrollment sources**. Select one recording, or use Ctrl/Shift to
   select several. Playback previews the first selection; deletion and model
   registration still require exactly one selected recording.
3. Choose **Export selected sources**, select a `.mphsources` destination outside
   the source library, and confirm the personal-data notice. Wait for completion.
   Export replaces an existing destination only after the complete new archive
   has been verified and written successfully.
4. Transfer the file through a protected channel. It is not encrypted.
5. On B, choose **Import sources** and the archive. The **People** and
   **Recordings** tabs show identity mappings, source UUIDs, passes, frames, size,
   preprocessing profile, and duplicate/conflict status after full verification.
6. Resolve person conflicts, then confirm **Import**. Preview alone does not
   register people, publish recordings, or create embeddings.
7. Close the dialog. Imported records appear in the source list and can be played
   back immediately. If B has no model features, install target bundles and use
   **Register to models** to create independent galleries from these sources.

To move both saved sources and existing features, transfer `.mphsources` and
`.mphgallery` separately, in either order. They share portable person UIDs;
consistent mappings must be used. Installing the same model/checkpoint and
preprocessing configuration is required to use old embeddings, but not to play
back foreground points. Model bundles, thresholds, and app preferences are not
transferred by Stage D.

Source UUIDs and session/pass IDs are preserved. Stage C therefore skips passes
already represented by compatible Gallery entries, including inactive entries.
Different target model contracts produce separate embeddings; source import
does not make embeddings interchangeable between models.

## Identity and Duplicate Rules

| Preview status | Behavior |
|---|---|
| New person | Proposes the archived readable ID if unused; `New ID` can change it. |
| Linked person | Reuses its existing portable UID mapping, regardless of name changes. Cannot be reassigned. |
| Readable ID conflict | Defaults to Skip. Choose a new unused ID or explicitly confirm `Merge into...` an existing person. |
| Previously deleted owner | Defaults to Skip. Enable restoration and choose an unused ID; a reused ID belonging to another person is never silently reclaimed. |
| Existing identical source UUID and portable manifest | Skips copying after checking local frame integrity. |
| Same UUID with different manifest, or same content fingerprint under another UUID | Blocks that person's import. Inspect the conflict; do not rename/relabel archives to bypass it. |

`Include` reverses Skip for eligible new/linked owners; restoring an occupied
deleted ID requires `New ID`. Mapping controls affect all sources for that owner.
An explicit merge preserves the destination person's current name, note, and
active/inactive state. No existing Gallery entry is deleted or reactivated.

Deleting a source locally and importing a previously exported archive restores
its saved points under the same UUID. Deleting its owner is different: explicit
owner restoration is required. Export of an orphan source is blocked until its
original owner is restored; creating another person with the same readable ID
does not restore ownership. This is migration/restore, not two-way synchronization
of deletions, names, or enrollment activity.

## Archive and Local Layout

```text
enrollment-sources.mphsources   # ZIP, schema mph-source-transfer-v1
  manifest.json
  sources/<source-uuid>/frame_000000.npy
  sources/<source-uuid>/...

data/
  gallery.sqlite3
  gallery.sqlite3_enrollment_sources/
    staging/transfer-<uuid>/    # temporary import; not visible as a recording
    records/<source-uuid>/
      manifest.json            # mph-foreground-source-v1
      frame_000000.npy
  source_transfer_backups/gallery-<unique-id>.sqlite3
```

The portable manifest retains identity UID, current exported name/ID/status,
captured name/ID, source creation time, session/pass IDs, source-clock timestamps,
frame order and segment markers, array SHA256/size, coordinate/unit/storage
convention, available direction/quality summaries, opt-in consent provenance,
and detector/filter/profile/source-hash settings needed to describe foreground
extraction. It excludes workstation paths, device serials, model configuration,
original local embedding IDs, per-window encoder output, and arbitrary metadata.
It is not a complete archive of sensor calibration or the original review log.
Capture consent provenance does not authorize sharing with a new recipient.

Imported arrays remain byte-for-byte identical. The sanitized local manifest has
its own SHA256; it can differ from the original capture manifest because local
paths and model-specific metadata are omitted. Source identity is the UUID, not
the folder's absolute path or the rewritten manifest hash.

SQLite adds `enrollment_source_transfers`, containing committed import time,
archive SHA256, counts, source IDs, restoration count, and backup path. Existing
source, Gallery, and model definitions are not rewritten. The incoming archive
is not copied to another permanent local archive; retain the original if needed.

## Integrity, Limits, and Recovery

- Preview checks every frame's SHA256, byte size, bounded NPY header, shape,
  float32 dtype, and finite coordinates. Object arrays/pickle, unsafe paths,
  symlinks, duplicate members/JSON keys, unknown members, and unsupported schemas
  are rejected. Files are never extracted with an unrestricted ZIP extractor.
- Import rechecks the archive hash and identity/source database snapshot. If A's
  archive or B's records changed since preview, close and preview again.
- Limits: 1,000 sources, 100,001 ZIP members, 64 MiB manifest, 32 GiB compressed
  and total expanded package, and 2,000,000 points per frame. Large selections
  should be split across several archives. One frame is processed at a time.
- UI transfers use `realtime.foreground_storage.library_limit_bytes` and
  `min_free_bytes` in `mph_gait_id/configs/system.yaml` (defaults 20 GiB and
  512 MiB). Import reserves room for new source files and a database backup;
  export conservatively reserves the uncompressed size plus overhead.
- The source writer lease prevents simultaneous capture, source deletion,
  cleanup, encoding, or another transfer. Close/cancel waits for the background
  operation to finish cleaning up; do not force-quit during a write.
- Before commit, failure/cancellation rolls back new database records and
  removes unpublished files. A successful commit is not undone by a late cancel.
  After a crash, close all app instances and use **Clean uncommitted files** to
  remove staging/orphan files. Do not manually merge SQLite files.
- Missing/corrupt files for an existing source cause an error, not a silent
  duplicate skip. After preserving evidence/backups, delete that damaged source
  through the UI and restore it from a verified archive.
- The pre-import SQLite backup contains no NPY arrays. It is not a full-system
  restore package. For administrative recovery, stop the app and back up the
  entire database plus its source-library directory and custom bundles together.

Archives and backups contain biometric and identity data. SHA256 detects
corruption, **not sender authenticity**; only import trusted packages. Protect
devices, transfer channels, recipients, and retention schedules. `.mphsources`,
source folders, and transfer backups are Git-ignored, not safe to publish.

## Verification

```bash
python -m unittest mph_gait_id.tests.test_source_transfer mph_gait_id.tests.test_source_transfer_ui -v
python -m unittest discover -s mph_gait_id/tests
python scripts/smoke_source_registration.py --device cpu --source-transfer
```

The smoke test uses synthetic foreground points and bundled PointNet-TMax,
MPH-Gait, and LidarGait++ Final-24 checkpoints. It checks exact source transfer,
live-window descriptor equivalence, and Gallery/source import interoperability.
It does not use participant recordings or measure recognition accuracy.

Before merge, test on the receiving Windows installation: multi-source export;
both UI languages/scaling; import into empty storage; playback; Stage C encoding
with multiple models; repeated import; same ID/different person; source deletion
and restore; explicit deleted-owner restore; cancel during large transfer; and
restarting the app. Server tests do not replace filesystem/GUI/hardware checks.
