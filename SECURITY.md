# Security and Local Data Handling

## Enrollment Foreground Sources

Opt-in enrollment recordings are biometric data. Their variable-size XYZ arrays,
owner metadata and temporary files are stored locally, are not encrypted by the
app, and must not be committed to Git. Deleting Gallery embeddings does not delete
these independent source recordings or previous backups. See
[source retention and deletion](docs/ENROLLMENT_SOURCES.md#deletion-transfer-and-privacy).

## Checkpoints Are Trusted Inputs

The application is a local research prototype, not a sandbox for unknown model
files. Gait imports, inference, and the Final-24 export helper use
`torch.load(weights_only=True)`, require a state dictionary of finite dense
tensors, and never fall back to unrestricted loading. Unsupported serialized
Python objects are rejected. Only import weights from a trusted source;
restricted loading is not a resource-exhaustion sandbox.

Ultralytics uses executable Python checkpoint objects internally. The app permits
only the hash-pinned YOLOv8n and YOLOv8n-seg assets, verified before constructing
YOLO. Renaming a different checkpoint does not bypass this check. Downloads are
verified as bytes before replacing an asset; downloading no longer loads a model.
SAM remains restricted to the pinned official ViT-B checkpoint. Supporting other
detectors requires an explicit asset and compatibility review.

Bundled assets have SHA256 values in `mph_gait_id/assets/manifest.yaml` and
bundle metadata. Verify them with:

```bash
python mph_gait_id/scripts/download_assets.py --profile realtime-yolo --check
```

A matching checksum verifies file identity against a trusted reference; it
does not make arbitrary pickle content safe. Computing a hash after importing
a file is not a security check on that file's origin. Do not expose the model
importer as a network service accepting untrusted uploads.

## Dependency Limitations

The required PyTorch range is `>=2.6,<3.0`; the loader rejects older versions
before deserialization. This removes the former range affected by
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
That advisory reports a restricted-loading bypass in versions through 2.5.1,
fixed for that issue in 2.6.0. Simply changing `weights_only` in an affected
version is not an adequate mitigation, and upgrading PyTorch does not make
unrestricted pickle loading safe.

The detector package is pinned to Ultralytics 8.3.221. Six gait checkpoints and
the bundled detector weights have CPU compatibility checks; CUDA and physical
Kinect capture still require local validation. The minimum PyTorch version
addresses the cited issue, not every possible present or future advisory.
Keep the environment patched, keep inputs trusted, and review advisories for
all installed dependencies before deployment.

## Embedding Compatibility

New Gallery keys include coordinate convention, input configuration and model
configuration hashes. Recorded encoder specifications are preserved on export.
Legacy compatibility is limited to the audited v0.3.0 encoder and the tested
migration target. Unknown definitions fail closed; no descriptors are deleted.
See [migration instructions](docs/INFERENCE_HARDENING.md). Neither a matching
feature contract nor temporal stability establishes calibrated biometric accuracy.

## Gallery and Captured Data

The repository starts with an empty gallery. Enrollment writes identity
metadata and descriptors to local SQLite storage. Gallery records and session
files are not encrypted by the application; use operating-system permissions
and appropriate storage protection for biometric data.

Keep generated content private:

- `data/`, including gallery databases and live enrollment sessions.
- `outputs/`, including similarity logs and performance reports.
- Captured RGB, depth, point clouds, and replay recordings.
- User-imported `mph_gait_id/model_bundles/user_*/` directories; their metadata
  may contain original local file paths and user-chosen names.

SQLite WAL, SHM, and journal files can contain database content even when the
main database is excluded. Numerical reports can still include identity IDs,
device serial numbers, or local paths; absence of images does not imply
anonymity. Redact reports before sharing and obtain appropriate participant
consent before collecting or distributing biometric data.

Unknown-person rejection is configurable but not independently calibrated.
Do not treat the prototype as a validated access-control or safety system.

## Version Control Boundaries

The ignore rules help prevent accidental additions of credentials, private
keys, generated data, imported bundles, and archives. Explicit exceptions keep
the eight approved gait/YOLO checkpoint assets trackable. Verify staged changes
before committing; `.gitignore` does not protect already tracked files or
remove content from previous commits.

If a credential is exposed, revoke or rotate it promptly. Removing a file or
its documentation link from the current branch does not remove old copies
from Git history, forks, or clones.

## Reporting a Security Concern

Contact [dennis.y.lai@gmail.com](mailto:dennis.y.lai@gmail.com) privately with
the affected commit, component, dependency versions, and a minimal description.
Do not attach real credentials, participant recordings, or gallery databases
to public issues or pull requests.
