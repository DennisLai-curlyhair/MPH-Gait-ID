# Repository Release Guide

[Documentation index](README.md) | [Traditional Chinese](RELEASE_GUIDE_zh.md)

## Files to Include

```text
Python source, tests, and scripts
mph_gait_id/configs/system.yaml
Model bundle YAML and approved bundled checkpoints
mph_gait_id/assets/manifest.yaml with SHA256 checksums
README, design, operation, and validation documentation
Dependency and packaging specifications
Third-party notices
```

Use [Final-24 deployment weights](FINAL24_WEIGHTS.md) to distinguish deployment
bundles from the Fixed-Special5 research bundles. Keep each bundle's checkpoint
hash and input metadata intact.

## Generated and Excluded Files

The repository's `.gitignore` excludes generated or machine-specific files,
including:

```text
Gallery SQLite databases
Run outputs, live enrollment sessions, and caches
Virtual environments, IDE files, and Python bytecode
The optional SAM ViT-B checkpoint
```

Distribute an empty gallery. Do not include registered identities, biometric
captures, RGB/depth recordings, similarity logs, or local runtime reports.
Ignoring a path does not remove files already tracked by Git; inspect the
staged changes before committing.

## Externally Hosted Assets

Large optional models can be distributed through release assets or a separate
file service. Preserve each asset's relative `target`, `release_name`, and
`sha256` in `mph_gait_id/assets/manifest.yaml`.

For assets supplied by the project's release provider, configure a base URL
that serves the declared release filenames:

```bash
export GAIT_IDENTITY_ASSET_BASE_URL="https://example.org/assets"
python mph_gait_id/scripts/download_assets.py --profile realtime-yolo
```

The example URL is illustrative. A shared-folder web page is not necessarily
a valid binary-download base URL. Other providers, such as Ultralytics and the
optional SAM URL, follow their manifest definitions.

## Pre-Release Checks

Run from the repository root after installing dependencies:

```bash
python -m mph_gait_id.ui_app --check-only
python mph_gait_id/scripts/doctor.py
python mph_gait_id/scripts/download_assets.py --profile realtime-yolo --check
python -m pytest mph_gait_id/tests
python scripts/validate_release.py
```

The `realtime-yolo` profile checks the bundled gait and YOLO assets without
requiring optional SAM weights. For a local SAM installation, additionally run:

```bash
python mph_gait_id/scripts/download_assets.py --profile sam --check
```

`--profile all --check` includes SAM and reports it missing when it has not
been installed. SAM is not required for the default release and must not be
included in the published repository. The release validator expects clean
runtime directories and no optional SAM checkpoint in the release tree.

Follow the [project README](../README.md) to verify Kinect on a clean Windows
machine with the Sensor SDK and real-time dependencies installed. An explicit
device check is:

```bash
python mph_gait_id/scripts/doctor.py --realtime --kinect
```

Report replay validation and hardware validation separately. Passing replay
tests does not establish Azure Kinect capture, calibration alignment, or live
hardware performance.

## License and Third-Party Status

Project license: **TBD**

Third-party model and weight redistribution status: **TBD**

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and
[LICENSE_PENDING.md](../LICENSE_PENDING.md). Third-party code and weights
remain subject to their respective ownership and terms.
