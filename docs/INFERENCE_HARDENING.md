# Inference Hardening and Gallery Upgrade

This Conference application branch fixes seven issues identified after v0.3.0.
It does not change gait model architectures, weights, research splits or training.

## Changes

| Area | Behavior |
|---|---|
| Checkpoint loading | Restricted state-dictionary loading, finite tensor validation, PyTorch >=2.6, no unsafe fallback. YOLO assets are hash-verified before loading. |
| Person segmentation | Request original-image masks and remove letterbox padding before any fallback resize. |
| Gallery compatibility | Version-2 keys include coordinate adapter, input channels, input/model config hashes, and existing T/checkpoint/profile fields. Nested LidarGait++ heads yield 7,936 dimensions from startup. |
| Identity stability | A passing window is provisional. Final `accepted` is true only after the configured stability requirement. UI shows Pending while accumulating. |
| Offline data quality | Reject empty/malformed/degenerate frames and incomplete windows; never replace empty clouds with zeros or pad a short sequence for enrollment. |
| Duplicate sources | Inactive features still reserve source ownership. Duplicate and ownership checks run again inside a SQLite writer transaction. |
| Operation reports | JSON writes are atomic. Report failure returns a warning without claiming that a committed Gallery enrollment failed. |

## Installation

Use Python 3.10--3.12 in an isolated environment. Install a matched PyTorch and
torchvision pair for the local CPU/CUDA setup using the
[official installer](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e ".[realtime]"
python mph_gait_id/scripts/doctor.py --realtime
python mph_gait_id/scripts/download_assets.py --profile realtime-yolo --check
```

Windows can use `requirements-windows.txt`; it now shares the supported runtime
requirements rather than pinning PyTorch 2.0/CUDA 11.7. Install Azure Kinect Sensor
SDK separately. Do not copy an old virtual environment between computers.

Gait checkpoint contents must be a tensor state dictionary under `model`, with
plain configuration metadata. All six distributed gait checkpoints satisfy this
format. Arbitrary pickled modules and custom YOLO checkpoints are not accepted.
The bundled YOLO weights are unchanged; Ultralytics is pinned to 8.3.221.

## Existing Gallery and Sources

Close **all** app instances, commit or abandon pending enrollment reviews, and
back up the complete data directory and custom bundles. Preserve the SQLite file
and its matching `<database filename>_enrollment_sources/` directory together.
Do not merge different computers' SQLite files by overwriting them.

New contracts use new keys. Old descriptors are not silently relabeled, deleted
or activated. Preview the migration:

```bash
python -m mph_gait_id.gallery_migration --db data/gallery.sqlite3
```

This preview is read-only. Entries with saved v0.3.0 encoder specifications can
be migrated only when the checkpoint, coordinate convention, model, point input,
window length and preprocessing profile match. When all entries are ready:

```bash
python -m mph_gait_id.gallery_migration --db data/gallery.sqlite3 --apply
```

The command rechecks the plan under a writer lock and backs up SQLite before
changing any rows. Embedding bytes, IDs, owners, inactive states and transfer
identities are preserved. Migration provenance is recorded in the database.
Running the command again is a no-op. Existing destination keys cause a refusal,
not an automatic merge. Stop and resolve blocked entries before applying.

### Missing Historical Metadata

Some ordinary v0.3.0 enrollments did not save the full encoder specification.
Their old keys alone cannot prove which coordinate adapter was used. Choose:

1. Re-encode retained foreground sources into the desired models using the
   source library. Old descriptors remain under their old keys.
2. Only if a distributed bundle was used with its **unmodified v0.3.0 settings**,
   explicitly confirm that bundle. Repeat the option for each verified bundle:

```bash
python -m mph_gait_id.gallery_migration --db data/gallery.sqlite3 \
  --confirm-legacy-bundle mph_gait_final24_len15_seed2
```

Preview first; add `--apply` only after checking the report. Confirmation is an
operator assertion about historical settings, not evidence inferred from the
database. It is unavailable for arbitrary custom weights. Modified or unknown
encoders require re-encoding/re-enrollment.

### Portable Archives

Previously exported v0.3.0 archives with a matching saved specification remain
importable using the normal Gallery preview/import UI. Imports use the new
contract after verification. Archives lacking provenance remain retainable but
unavailable for recognition; use the original database's migration or sources.
New exports preserve the stored specification even if local manifests change.

Foreground extraction settings are distinct from the encoder contract. Saved
foreground sources cannot be re-segmented without original RGB/depth observations.
This upgrade fixes mask alignment but does not repair old masks or descriptors.
For live deployments, capture a new validation pass and consider fresh enrollment
if old recordings contain background points or miss parts of the person. Use a
distinct preprocessing profile for intentionally different foreground pipelines.

## Recognition and Registration

- With stability=3, the first two passing windows remain provisional; three
  accepted windows of the same identity are needed for a stable decision.
  A rejected or different-identity window breaks that full-window agreement.
- `window_accepted` records the score/margin decision; `accepted` is the final
  stabilized decision. This is not an open-set calibration claim.
- Inactive sources cannot be assigned to another person. Same-person explicit
  duplicate overrides do not authorize changing ownership.
- Offline folders require at least T real frames after the configured initial
  drop. Every frame requires numeric XYZ with at least two distinct finite points.
  This is a structural validity floor, not a sufficient human-motion quality test.
- A `report_warning` means the operation completed but its JSON report failed.
  Gallery enrollment remains committed; do not repeat it just to regenerate JSON.

## Validation

```bash
python -m unittest discover -s mph_gait_id/tests
python -m unittest discover -s scripts/tests
python scripts/smoke_source_registration.py --device cpu
python scripts/validate_release.py
```

Server checks include synthetic landscape/portrait masks, unsafe checkpoint
rejection, duplicate ownership, transaction-time rechecks, report failures,
legacy migration, and actual three-model source registration/transfer.
All six gait outputs were also compared against the v0.3.0 adapter using the
same synthetic valid input and PyTorch 2.10.0 CPU: maximum absolute difference
was zero for every checkpoint. This equivalence does not cover arbitrary forks,
custom architectures, invalid/short inputs, GPU kernels or mask generation.

Local acceptance still requires Azure Kinect capture, Windows/CUDA dependencies,
RGB-mask-point alignment, guided enrollment, Pending/Stable transitions, and a
real-data export/import round trip. Backups and source recordings remain private
biometric data. See [SECURITY.md](../SECURITY.md).
