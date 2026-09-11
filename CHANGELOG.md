# Changelog

## Unreleased

### UI
- Add a Realtime readiness checklist, localized pipeline states, frame-buffer
  progress and guided enrollment phase indicators.
- Wait for capture readiness and pass-command acknowledgement; lock configuration
  through pending review and worker cleanup. Run device probes and Stop in the
  background, and preserve settings when controls are released.
- Retain review selections on reopen, disable empty/duplicate commits and show
  save progress; retain failed reviews for retry.
- Separate multi-model job results from logs, clarify transfer phases and check
  offline Gallery availability before encoding.
- Add workflow regression tests and bilingual phase-two acceptance guides.
- Reserve bilingual control/tab widths and table-heading space; wrap long status
  text and add scrolling for narrow settings and preview tool panels.
- Add resizable Realtime sidebar and useful minimum areas for Offline previews
  and Performance results; keep benchmark actions outside scrolling settings.
- Switch languages without changing person data, model settings, or running jobs.
- Preserve source pass/frame/zoom and surviving selections on refresh; pause
  playback on tab exit without resetting position.
- Preserve Gallery person/pass selection and scroll position using logical pass
  identity, not row numbers. Revisit model pages without reapplying unchanged
  bundle defaults; changed checkpoints/configurations still refresh defaults.
- Add presentation/state regression tests and bilingual local acceptance guides.

### Added
- Stage D foreground-source export/import using versioned `.mphsources` archives,
  with multi-source selection, verified previews, explicit person mapping, and
  deleted-owner restoration. Gallery descriptor transfer remains separate.
- Preserved source UUIDs, session/pass boundaries, camera XYZ, and preprocessing
  provenance for playback and Stage C registration on another installation.
- Cancellable background transfers, bounded data-only archive validation,
  pre-import SQLite backups, atomic database commits, and import audit records.
- English/Traditional Chinese source-transfer controls and migration guides.
- Source-transfer regression tests and a real-checkpoint three-model smoke-test
  option using synthetic recordings.

### Compatibility
- Existing recordings and embeddings are retained; no model weights, inference
  contract hashes, dependencies, or release version are changed.
- Transfer archives are not encrypted and contain personal biometric data.
  `.mphsources` and source-transfer backups are excluded from Git.
- See [Source transfer](docs/SOURCE_TRANSFER.md) for local acceptance checks.

## 0.4.0

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
- Added the missing einops dependency required by LidarGait++ and its environment check.
- Fixed English/Traditional Chinese switching for Offline titles and Performance
  settings, section titles, controls, headings, and fixed status messages.
- Fixed enrollment-source playback scheduling by removing the registration
  action's name collision with Tkinter's internal callback registration method.

### Installation
- The Windows preset and both Windows bootstrap profiles explicitly install
  torch 2.10.0+cu126 and torchvision 0.25.0+cu126. Missing GPU wheels cause an
  installation error rather than a CPU-only fallback.
- CUDA execution still requires a compatible NVIDIA GPU and driver. Generic
  requirements remain platform-neutral; Python 3.10--3.12 is supported.

### Upgrade
- The Windows PyTorch 2.0/CUDA 11.7 preset is retired. Use the updated Windows
  preset in a fresh environment; Ultralytics remains pinned to 8.3.221.
- Back up data and preview the explicit Gallery contract migration before use.
  No model weights or source recordings are rewritten. See
  [Inference hardening](docs/INFERENCE_HARDENING.md) for migration and local tests.

### Validation
- 177 automated application tests passed; five display-dependent GUI tests
  were skipped on the server. Playback callbacks were tested with real Tcl timers.
- Local application testing was completed successfully before release preparation.

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
