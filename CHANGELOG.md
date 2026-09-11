# Changelog

## Unreleased

### Fixed
- Restricted gait checkpoint loading with a patched PyTorch minimum; verified
  YOLO assets before deserialization and checked downloads before replacement.
- YOLO masks now use original-image coordinates with letterbox-aware fallback.
- Versioned coordinate/input/model Gallery contracts and configuration-aware
  runtime caching; correct nested LidarGait++ descriptor dimensions at startup.
- Provisional identity matches no longer count as accepted before stability.
- Offline enrollment rejects invalid point frames and incomplete windows.
- Inactive source ownership and duplicate checks are enforced in the write transaction.
- Failed JSON report writes no longer report committed enrollments as failures.

### Upgrade
- The Windows PyTorch 2.0/CUDA 11.7 preset is retired; use PyTorch >=2.6,<3.0 and
  the pinned Ultralytics 8.3.221 runtime with a locally matched torchvision build.
- Back up data and preview the explicit Gallery contract migration before use.
  No model weights or source recordings are rewritten. See
  [Inference hardening](docs/INFERENCE_HARDENING.md) for migration and local tests.

## 0.3.0

### Added
- Opt-in enrollment foreground XYZ recording before fixed-point sampling, with
  bounded background writing and storage limits.
- A source library with review-selected walking passes, playback, and independent
  recording deletion without removing existing Gallery descriptors.
- Multi-model registration from one saved recording to selected installed
  checkpoints, with per-model clip lengths, progress, and cancellation.
- Pass-level duplicate checks and local registration job audit records.
- English and Traditional Chinese controls and source-registration guides.
- A synthetic three-model smoke test for source encoding, live-output equivalence,
  and Gallery transfer.

### Fixed
- LidarGait++ Gallery transfer compatibility now reads the flattened descriptor
  dimension from the model's nested part heads.

### Safety
- All new target galleries in one registration job are committed atomically;
  cancellation or failure leaves existing descriptors and recordings unchanged.
- Source ownership, preprocessing profiles, checkpoint hashes, and model
  compatibility remain enforced. Reusing a Person ID does not reassign old sources.
- Previously registered passes are skipped, including inactive Gallery entries.
- Source files are hash-checked and protected by a library lock during reuse.
- Foreground recordings remain local. Gallery transfer archives contain
  descriptors only, not source recordings or source-registration job history.
- Model weights and the existing live encoding path are unchanged.

### Upgrade
- Stop the application and commit or abandon pending enrollment reviews first.
- Back up the complete local data directory before upgrading. When moving to a
  fresh checkout, copy the Gallery database and its matching source-library
  directory together; preserve custom model bundles separately.
- Existing saved source recordings can be reused without recapture. Gallery
  descriptors alone cannot reconstruct a source recording.
- Source and job tables are initialized automatically; no manual SQL migration
  is required. See the upgrade instructions in README.md.

## 0.2.0

### Added
- Portable Gallery export and import.
- Identity renaming and permanent fragment/person deletion.
- Explicit restoration of previously deleted Gallery records.
- Pre-operation backups and restoration history.

### Safety
- Restoring deleted identities requires unused target IDs.
- Model compatibility and duplicate checks remain enforced.
- Back up the Gallery database before upgrading.