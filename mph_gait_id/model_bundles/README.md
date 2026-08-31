# Model Bundles

A model bundle is the compatibility boundary used by MPH-Gait ID. Each bundle
contains a checkpoint and metadata describing the architecture and input
contract.

| Bundle | Model | Input | Default T |
|---|---|---|---:|
| `pointnet_tmax_fixed_special5_seed0_split0` | PointNet-TMax | axis-aligned XYZ | 15 |
| `mph_gait_fixed_special5_seed0_split0` | MPH-Gait | axis-aligned XYZ | 15 |
| `lidargaitpp_fixed_special5_seed0_split0` | adapted LidarGait++ | official adapter input | 15 |

Gallery descriptors are keyed by bundle ID, checkpoint hash, coordinate
adapter, preprocessing profile, and clip length. Switching to an incompatible
bundle never reuses existing gallery descriptors silently.

To add a compatible checkpoint, use the UI importer and select the matching
template bundle. To add a new architecture, implement a model adapter and
provide a new `bundle.yaml` rather than modifying an existing bundle.
