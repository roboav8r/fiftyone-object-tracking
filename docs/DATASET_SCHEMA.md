# Dataset schema

This plugin's operators are dataset-agnostic: they consume a *canonical*
grouped FiftyOne schema, regardless of which loader produced it. Source
datasets are produced by a separate, private dataloaders repo (one loader
per source format, each emitting this schema).

## What it expects on the source dataset

The `build_trajectories` operator consumes a grouped FiftyOne dataset
with:

| Field | On slice | Notes |
|---|---|---|
| `scene_name` | all | string id per scene |
| `frame_idx` | all | int ordinal within scene |
| `m_frame_timestamp` | all | seconds.fractional or pure-ns string |
| `world_to_base` | lidar | SE(3) as `{translation, quaternion_xyzw, matrix_4x4_row_major}` |
| `detections` | lidar (keyframes) | 3D cuboids: `location` (xyz base), `rotation` (`[rx, ry, rz]` XYZ-Euler), `dimensions` (lwh), `instance` (cross-frame link) |
| `detections.detections.tracking_id` | lidar (keyframes) | source-side identifier (stamped by the loader) |
| `detections.detections.segment_index` | lidar (keyframes) | source-side segment id (0 if not applicable) |

`info["ego_size_lwh_m"]` (optional but recommended) on the source
dataset's info is read for the ego's BEV footprint.

## Cuboid rotation convention

`fo.Detection.rotation` is `[rx, ry, rz]` — intrinsic rotation about
each axis. The plugin reads `rot[2]` as yaw (under XYZ Euler), and
the trajectory builder composes per-frame world rotations from the
full `world_from_base` quaternion. Source datasets MUST decompose
the source quaternion with `as_euler("XYZ")` (not `"zyx"`) — the
loader scripts in `fiftyone-tracking-loaders` do this correctly.
