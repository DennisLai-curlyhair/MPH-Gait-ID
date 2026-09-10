# MPH-Gait ID

MPH-Gait ID is a desktop application for gallery-based point-cloud gait
enrollment and identification. It supports offline sequence replay and
single-person real-time processing with Azure Kinect DK.

The repository is distributed with an empty gallery database. Enrollment data,
captured frames, similarity logs, and performance reports are created locally
and are excluded from version control.

## Research Reproducibility Notice

This repository is published to support academic evaluation and reproduction
of the MPH-Gait identification workflow. Third-party projects, source-derived
components, pretrained weights, names, and trademarks remain the property of
their respective owners and are governed by their original terms. This project
does not claim ownership of or grant additional rights to those materials. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the component inventory
and upstream sources.

## Features

- Offline enrollment and identification from point-cloud sequence folders.
- Real-time Azure Kinect DK device discovery and capture.
- RGB person segmentation followed by calibrated foreground-point extraction.
- Point-cloud filtering, 1,024-point sampling, and configurable rolling windows.
- PointNet-TMax, MPH-Gait, and official-code adapted LidarGait++ model bundles.
- Guided multi-pass enrollment with review, resume, and commit controls.
- Gallery management with model/checkpoint/preprocessing compatibility keys.
- [Person rename and scoped permanent deletion](docs/GALLERY_MANAGEMENT.md),
  with pre-edit backups and protection against reimporting deleted identities.
- [Portable Gallery export/import](docs/GALLERY_TRANSFER.md) with preview,
  explicit identity-conflict resolution, model checks, deduplication, and backups
  ([繁體中文](docs/GALLERY_TRANSFER_zh.md)).
- [Opt-in enrollment foreground source recording](docs/ENROLLMENT_SOURCES.md),
  pass-level review, playback, storage limits, and independent source deletion
  ([繁體中文](docs/ENROLLMENT_SOURCES_zh.md)).
- [Multi-model registration from saved foreground sources](docs/SOURCE_REGISTRATION.md),
  with per-model window lengths, progress, cancellation, and atomic Gallery writes.
- Rank-based identity matching with configurable unknown-score, margin, and
  temporal-stability thresholds.
- RGB, foreground point-cloud, FPS, predicted identity, and similarity displays.
- English and Traditional Chinese UI resources.
- Replay and pipeline validation tools for development without Kinect hardware.

The current real-time workflow selects one primary person. It is not a
multi-person tracking system.

## Processing Pipeline

```text
Azure Kinect RGB + depth + organized XYZ
                  │
                  ▼
        person segmentation mask
                  │
                  ▼
 calibrated RGB/depth/XYZ foreground mapping
                  │
                  ▼
 valid-depth, median/MAD, and robust outlier filtering
                  │
                  ▼
       1,024-point sampling per frame
                  │
                  ▼
       rolling T-frame gait window
                  │
                  ▼
       selected gait model descriptor
                  │
                  ▼
 gallery similarity aggregation and unknown rejection
                  │
                  ▼
       identity, score, FPS, and visual feedback
```

## Repository Layout

```text
MPH-Gait-ID/
├── mph_gait_id/
│   ├── configs/           # application defaults
│   ├── model_bundles/     # model metadata and bundled gait checkpoints
│   ├── model_weights/     # bundled YOLO detector weights
│   ├── models/            # deployable model architectures
│   ├── realtime/          # device, detector, preprocessing, enrollment pipeline
│   ├── scripts/           # bootstrap, doctor, replay, validation, asset tools
│   ├── tests/             # core unit tests
│   └── ui/                # Tk desktop interface
├── data/                  # generated gallery and enrollment sessions
├── outputs/               # generated run and benchmark reports
├── docs/
├── pyproject.toml
├── requirements.txt
├── requirements-realtime.txt
└── run.py
```

## Bundled Models and Assets

| Asset | Included | Approximate size |
|---|---:|---:|
| PointNet-TMax, Fixed-Special5 seed0 split0 | yes | 3.8 MB |
| MPH-Gait, Fixed-Special5 seed0 split0 | yes | 4.9 MB |
| Adapted LidarGait++, Fixed-Special5 seed0 split0 | yes | 19.7 MB |
| PointNet-TMax, Final-24 len15 seed2 | yes | 1.28 MB |
| MPH-Gait, Final-24 len15 seed2 | yes | 1.66 MB |
| Adapted LidarGait++, Final-24 len15 seed2 | yes | 10.21 MB |
| YOLOv8n detector | yes | 6.5 MB |
| YOLOv8n-seg detector | yes | 7.1 MB |
| SAM ViT-B checkpoint | no, optional download | 375 MB |

All included files are verified against
`mph_gait_id/assets/manifest.yaml`. The SAM checkpoint is optional because
YOLO segmentation is the default and lower-cost foreground extractor.

The Final-24 bundles use 24 training identities and retain the final checkpoint
of the prescribed budget: 50 epochs for PointNet-TMax and MPH-Gait, and 10,000
iterations for adapted LidarGait++. These are inference-only exports with
unchanged model tensors; optimizer states and local training paths are omitted.
Seed 2 is a deployment release choice, not a claim that it is the best seed for
every model or endpoint. See [Final-24 deployment weights](docs/FINAL24_WEIGHTS.md).

Select a `Final-24 len15 seed2` entry in the model selector to use the new
weights. The existing default bundle is unchanged. Final-24 and Fixed-Special5
descriptors are incompatible: enroll identities again under the selected
Final-24 bundle. Existing gallery entries are retained for their original model.

## Requirements

Only import trusted checkpoints. The current loader uses unrestricted PyTorch
pickle loading, and the historical dependency range requires security review.
See [Security and local data handling](SECURITY.md) before using custom weights
or deploying the application.

### Core Offline Application

- Python 3.10 or 3.11
- Tkinter
- PyTorch 2.x
- NumPy, OpenCV, Pillow, and PyYAML

On Ubuntu/Debian, Tkinter may require:

```bash
sudo apt-get install python3-tk
```

Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python mph_gait_id/scripts/doctor.py
python mph_gait_id/scripts/download_assets.py --profile offline-demo --check
```

The gait checkpoints and YOLO weights are already present in this package, so
the checks should not download them.

### Azure Kinect Real-Time Mode

Install the Azure Kinect Sensor SDK at the operating-system level before
installing PyK4A. Then install real-time dependencies:

```bash
python -m pip install -r requirements-realtime.txt
python mph_gait_id/scripts/doctor.py --realtime
python mph_gait_id/scripts/doctor.py --realtime --kinect
```

The `--kinect` check opens the device and verifies capture access; USB
enumeration alone is not treated as a successful hardware test.

For the optional SAM detector:

```bash
python -m pip install -e '.[sam]'
python mph_gait_id/scripts/download_assets.py --profile sam
python mph_gait_id/scripts/doctor.py --sam
```

### Windows Bootstrap

Install Azure Kinect Sensor SDK 1.4.1 first, then run PowerShell from the
repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\mph_gait_id\scripts\bootstrap_windows.ps1 -Profile realtime
.\mph_gait_id\scripts\run_ui_windows.ps1
```

The pinned Windows preset is stored in `requirements-windows.txt`. A different
PyTorch wheel may be required for newer GPU drivers.

## Start the Application

After installation, use any of the following:

```bash
mph-gait-id
python -m mph_gait_id.ui_app
python run.py
bash mph_gait_id/scripts/run_ui.sh
```

Validate package paths and model bundles without opening the UI:

```bash
python -m mph_gait_id.ui_app --check-only
python mph_gait_id/scripts/validate_model_bundles.py
python scripts/validate_release.py
```

## Enrollment Workflow

1. Select a model bundle and frame length compatible with its metadata.
2. Choose offline input or connect Azure Kinect DK.
3. Enter a stable person ID and optional display name.
4. Start a guided enrollment session.
5. Capture one or more walking passes.
6. Review accepted windows, resume capture if necessary, then commit.
7. The application stores normalized descriptors in `data/gallery.sqlite3`.

Gallery entries are separated by bundle ID, checkpoint SHA256, coordinate
adapter, preprocessing profile, and clip length. Descriptors produced by
incompatible models or preprocessing settings are not mixed.

## Multi-Model Enrollment from Saved Sources

After saving a foreground recording during enrollment, open **Enrollment
sources**, select it, and choose **Register to models**. Select its passes and
the installed target checkpoints, set T per model, and start one sequential
encoding job. Each model gets its own compatible Gallery embeddings; the
original recording and existing embeddings are retained. Registered passes are
skipped. Cancellation or failure adds no partial multi-model galleries.

See the [Stage C guide](docs/SOURCE_REGISTRATION.md)
([繁體中文](docs/SOURCE_REGISTRATION_zh.md)) for window rules, safeguards, and testing.

## Identification Workflow

Each complete probe window is embedded and compared with all compatible
gallery descriptors. Similarities are aggregated per identity using the
configured top-k rule and then across probe windows. The highest identity score
is returned only when it satisfies the configured score, top-1/top-2 margin,
and temporal-stability rules.

The default unknown thresholds are provisional operating values. Open-set use
requires calibration with known and unknown identities that are independent of
the enrolled gallery and model-development data.

## Model Bundles and Custom Checkpoints

Each directory below `mph_gait_id/model_bundles/` contains:

```text
bundle.yaml
<checkpoint filename declared in bundle.yaml>
```

`bundle.yaml` records architecture, input channels, coordinate convention,
point count, clip length, preprocessing profile, training protocol, and
checkpoint SHA256. Use the UI checkpoint importer with a compatible template
bundle when replacing a weight file. Changing architecture or input semantics
requires a new bundle definition and adapter; renaming a checkpoint is not
sufficient.

Large future release weights can be hosted separately by setting
`GAIT_IDENTITY_ASSET_BASE_URL` and using:

```bash
python mph_gait_id/scripts/download_assets.py --profile offline-demo
```

Verify only the included Final-24 weights without downloading:

```bash
python mph_gait_id/scripts/download_assets.py --profile final24 --check
```

## Validation and Tests

Install the test dependencies, then run:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q mph_gait_id
python -m pytest mph_gait_id/tests
python mph_gait_id/scripts/validate_model_bundles.py
python scripts/validate_release.py
python mph_gait_id/scripts/validate_live_pipeline.py
python mph_gait_id/scripts/validate_guided_enrollment.py
python mph_gait_id/scripts/validate_enrollment_review_resume.py
```

Replay tools accept synchronized RGB/depth/XYZ folders and allow pipeline
testing without an attached Kinect device. See the [UI guide](docs/UI_GUIDE.md)
and [system design](docs/DESIGN.md) for the interface and system boundaries.

## Documentation

English and Traditional Chinese guides are listed in the
[documentation index](docs/README.md).

| Guide | English | Traditional Chinese |
|---|---|---|
| Interface and enrollment | [UI guide](docs/UI_GUIDE.md) | [UI guide](docs/UI_GUIDE_zh.md) |
| System architecture | [Design](docs/DESIGN.md) | [Design](docs/DESIGN_zh.md) |
| Live efficiency measurement | [Benchmark](docs/PERFORMANCE_BENCHMARK_GUIDE.md) | [Benchmark](docs/PERFORMANCE_BENCHMARK_GUIDE_zh.md) |

For included deployment checkpoints and re-enrollment requirements, see
[Final-24 deployment weights](docs/FINAL24_WEIGHTS.md). Security reporting and
trusted-input requirements are in [SECURITY.md](SECURITY.md).

<!-- ## Privacy and Generated Data

Do not commit `data/`, `outputs/`, captured RGB/depth frames, person
identifiers, or gallery databases. Obtain the required participant consent and
apply the appropriate retention policy before collecting biometric data. -->

## Citation and License

Publication citation metadata: **TBD** (`CITATION.cff.template`).

Project license: **TBD** (`LICENSE_PENDING.md`). Third-party model and
software notices are listed in `THIRD_PARTY_NOTICES.md`.
