# Model Bundles

A model bundle is the compatibility boundary used by MPH-Gait ID. Each bundle
contains a checkpoint and metadata describing the architecture and input
contract.

| Bundle | Model | Input | Default T |
|---|---|---|---:|
| `pointnet_tmax_fixed_special5_seed0_split0` | PointNet-TMax | axis-aligned XYZ | 15 |
| `mph_gait_fixed_special5_seed0_split0` | MPH-Gait | axis-aligned XYZ | 15 |
| `lidargaitpp_fixed_special5_seed0_split0` | adapted LidarGait++ | official adapter input | 15 |
| `pointnet_tmax_final24_len15_seed2` | PointNet-TMax, Final-24 | axis-aligned XYZ | 15 |
| `mph_gait_final24_len15_seed2` | MPH-Gait, Final-24 | axis-aligned XYZ | 15 |
| `lidargaitpp_final24_len15_seed2` | adapted LidarGait++, Final-24 | official adapter input | 15 |

Final-24 bundles use descriptive checkpoint filenames matching the bundle ID,
rather than `checkpoint.pt`. The filename and SHA256 are recorded in
`bundle.yaml`. Their `fold: -1` distinguishes full-development-set deployment
training from rotating Fixed-Special5 splits. Each Final-24 bundle also contains
`provenance.json` with source checkpoint/config/protocol hashes.

See [Final-24 deployment weights](../../docs/FINAL24_WEIGHTS.md) for training
identities, export details, and enrollment requirements.

Gallery descriptors are keyed by bundle ID, checkpoint hash, coordinate
adapter, preprocessing profile, and clip length. Switching to an incompatible
bundle never reuses existing gallery descriptors silently.

To add a compatible checkpoint, use the UI importer and select the matching
template bundle. To add a new architecture, implement a model adapter and
provide a new `bundle.yaml` rather than modifying an existing bundle.
