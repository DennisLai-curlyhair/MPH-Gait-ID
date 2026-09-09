# Final-24 Deployment Weights

## Included Bundles

| UI model | Bundle ID / checkpoint stem | Training endpoint | File size |
|---|---|---|---:|
| PointNet-TMax Final-24 len15 seed2 | `pointnet_tmax_final24_len15_seed2` | Epoch 50 | 1.28 MB |
| MPH-Gait Final-24 len15 seed2 | `mph_gait_final24_len15_seed2` | Epoch 50 | 1.66 MB |
| LidarGait++ Final-24 len15 seed2 | `lidargaitpp_final24_len15_seed2` | Iteration 10,000 | 10.21 MB |

Each bundle resides under `mph_gait_id/model_bundles/<bundle_id>/` and contains:

```text
bundle.yaml
<bundle_id>.pt
provenance.json
```

The actual weight files are tracked in Git, not Git LFS pointers. No additional
download is required after cloning this version. Existing Fixed-Special5
seed-0/split-0 bundles are retained, and the application default is unchanged.

## Training and Release Selection

All three models use Final-24 protocol `final24_fixed5_eval_v1`, training seed
2, 15 frames, and 1,024 points per frame. PointNet-TMax and MPH-Gait use the final
50-epoch state; adapted LidarGait++ uses the final 10,000-iteration state.
There is no held-out validation checkpoint selection in Final-24.

The 24 training identities are:

```text
P001 P002 P004 P005 P006 P008 P009 P010 P011 P012 P013 P014
P015 P016 P017 P020 P021 P022 P023 P024 P025 P027 P028 P029
```

Training uses C2/C3 V001-V016. P003, P007, P018, P019, and P026 remain outside
training. Their C2/C3 V017-V020 sequences form the Final-24 evaluation gallery;
C1 and C4/C5/C7 are separate personal- and special-clothing probe endpoints.

Seed 2 is the publisher-selected deployment release. It is not presented as a
validation-selected or universally best seed: the historical results do not
rank every method and endpoint identically. These bundles do not replace the
conference five-split, three-seed evaluation or its checkpoints. Selecting a
deployment seed after inspecting stress results does not create a new unbiased
test result.

## Input and Model Compatibility

The application accepts Kinect camera XYZ in millimeters. The bundle selects
the adapter `[X,Y,Z] -> [Z,X,-Y] * 0.001` for forward/lateral/height in meters.
PointNet-TMax and MPH-Gait retain frame-wise centering without scale removal.
LidarGait++ retains centered, radius-normalized XYZ and an additional metric
height feature through the existing scale-aware adapter.

The exported LidarGait++ classification head has 24 classes, matching this
training run; the older split-0 bundle has 14. PointNet-TMax and MPH-Gait retain
the training implementation's 29-way head and dataset-wide identity mapping.
Only training identities supplied optimization samples. Classification heads
are not used to assign deployment identities; gallery matching performs that
step.

Final-24 metadata uses `fold: -1` because it is not a rotating evaluation fold.
It must not be relabeled as Fixed-Special5 split 0 merely because an original
training launcher used a single split index internally.

## Using the Weights

1. Update the repository and install its documented dependencies.
2. Verify the three checkpoints:

```bash
python mph_gait_id/scripts/download_assets.py --profile final24 --check
```

3. Start the application and select a `Final-24 len15 seed2` model entry.
4. Enroll identities under that bundle before identification.

Training identities also require gallery enrollment. Switching from a
Fixed-Special5 checkpoint to Final-24 changes the embedding space and the
compatibility key. Existing gallery entries remain stored but must not be
reused as Final-24 embeddings. No identities or sessions are included here.

The supplied clip length is 15. A different rolling-window length is an
explicit deployment setting, not a separately validated checkpoint. Similarity
thresholds remain provisional and require independent calibration after a
model change; the release does not certify open-set accuracy.

## Export and Provenance

Each checkpoint is an inference-only export containing the unchanged model
state, portable model/input configuration, and training-endpoint metadata.
Optimizer and scheduler states and machine-specific training paths are not
included. Smaller files therefore do not indicate quantization, pruning, or
retraining. State-dictionary keys, dtypes, and tensor values are checked against
the source before the export is accepted.

`bundle.yaml` and `mph_gait_id/assets/manifest.yaml` identify each exported
checkpoint by SHA256. `provenance.json` records hashes of the original
checkpoint, training configuration, protocol, and export script. Adapted
LidarGait++ retains the pinned OpenGait commit and third-party notices.

The original run identity is `20260817-182113_final24_len15_3seed`. Maintainers
with the trusted original training outputs can rebuild exports into a separate
directory:

```bash
python scripts/export_final24_bundles.py \
  --source-root /path/to/experiment_final24 \
  --run-id 20260817-182113_final24_len15_3seed \
  --seed 2 \
  --output-dir /path/to/new-bundles
```

The exporter refuses to overwrite an existing bundle and does not modify its
source runs or update asset hashes automatically. Serialized file hashes can
depend on the PyTorch version; rebuilt exports require their own checksum
verification even when tensor values are identical. Only load trusted original
training checkpoints, since those may contain Python pickle objects.

Full package checks:

```bash
python scripts/validate_release.py
python mph_gait_id/scripts/validate_model_bundles.py
python -m unittest discover -s mph_gait_id/tests
```

These checks verify assets and model execution. They do not replace an Azure
Kinect hardware test or a new real-world accuracy evaluation.
