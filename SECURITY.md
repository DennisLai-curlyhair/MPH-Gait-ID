# Security and Local Data Handling

## Checkpoints Are Trusted Inputs

The application is a local research prototype, not a sandbox for unknown model
files. Its current model store and adapter use unrestricted PyTorch checkpoint
loading (`weights_only=False`) for compatibility with research checkpoints.
Importing a malicious checkpoint can execute code before its architecture or
metadata is validated. Only import weights from a trusted source.

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

The current package range (`torch>=2.0,<2.6`) and pinned Windows preset
(`torch==2.0.1+cu117`) are historical compatibility settings, not a hardened
dependency baseline. They include versions affected by
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
That advisory reports a restricted-loading bypass in versions through 2.5.1,
fixed for that issue in 2.6.0. Simply changing `weights_only` in an affected
version is not an adequate mitigation, and upgrading PyTorch does not make
unrestricted pickle loading safe.

A secure-loader and dependency migration requires compatibility tests for all
bundled models, Ultralytics, CUDA, and Kinect capture. The current repository
does not claim this migration is complete. Keep inputs trusted and isolate the
research environment from credentials or other sensitive workloads. Review
current advisories for all installed dependencies before deployment.

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
