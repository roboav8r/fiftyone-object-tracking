# fiftyone-object-tracking-plugin

FiftyOne plugin for working with grouped 3D tracking datasets:

- **Build per-trajectory datasets** from a tracking source via the
  `build_trajectories` operator. Each trajectory sample is a Parquet
  payload + ~50 filter-friendly scalar facets grouped under
  `Identity` / `Coverage` / `Position (base)` / `Position (world)` /
  `Motion` / `Shape` / `QC`.
- **`BEVTrackVisualization` panel** — per-scene bird's-eye-view of
  object trajectories with a timeline scrubber, base ↔ world toggle,
  per-instance presence rows, and pan / zoom.
- **`TrajectoryRenderer` custom sample renderer** — renders per-cell
  BEV plots for the trajectories dataset's Parquet samples in the
  App grid + modal (start marked with **o**, end with **x**).

Source datasets are produced by a separate, private dataloaders repo
(one loader per source format, each emitting the canonical
per-(scene, frame, sensor) grouped schema). This plugin's
trajectory operator is dataset-agnostic — it consumes the canonical
schema, regardless of which loader produced it.

## Install

Into a FiftyOne Enterprise deployment:

```bash
fiftyone plugins download \
    https://github.com/roboav8r/fiftyone-object-tracking-plugin
```

For local development, symlink the repo into your plugins dir:

```bash
git clone https://github.com/roboav8r/fiftyone-object-tracking-plugin
cd fiftyone-object-tracking-plugin
./install.sh   # symlinks into ~/fiftyone/__plugins__/
```

The plugin's Python operators need `numpy`, `scipy`, `pandas`, and
`pyarrow` (see `requirements.txt`); install them in your FOE
deployment's Python environment.

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

## Usage

### Build trajectories from the App

Open the source tracking dataset → operator palette → "Build
trajectories dataset" → fill the form (target name, trajectory_root,
overwrite) → run. The new trajectories dataset appears in the App's
dataset selector.

### Build trajectories from the SDK / CLI

```bash
fiftyone operators execute \
    @roboav8r/fiftyone-object-tracking-toolkit/build_trajectories \
    --params '{"source": "<source-tracking>",
               "target": "<target-trajectories>",
               "trajectory_root": "gs://your-bucket/derived/trajectories",
               "overwrite": true}'
```

```python
import fiftyone.operators as foo
op = foo.get_operator("@roboav8r/fiftyone-object-tracking-toolkit/build_trajectories")
op({
    "source": "delivery-robot-tracking",
    "target": "delivery-robot-trajectories",
    "trajectory_root": "gs://your-bucket/derived/trajectories",
    "overwrite": True,
})
```

### Browse trajectories in the App

After build, open the trajectories dataset. The custom
`TrajectoryRenderer` draws a BEV thumbnail per cell (forward up,
left to the left, `o` at the trajectory start, `x` at the end). The
sidebar exposes ~50 filter facets in 7 groups, including the **QC**
group:

| Field | Type | What it surfaces |
|---|---|---|
| `n_distinct_classes` | Int | `> 1` flags multi-class tracks — usually an annotator-side ID collision rather than a genuine class transition |
| `tracking_names_distinct` | List[Str] | The actual class set (e.g. `["human.pedestrian", "vehicle"]`) |
| `max_step_jump_m` | Float | Largest world-frame XY jump between consecutive keyframes — flags physically-impossible teleports |
| `max_gap_s` | Float | Longest inter-keyframe gap in seconds (keyframe-only by construction) |

### Browse a single scene with the BEV panel

On the source tracking dataset (not the trajectories one), open the
`BEVTrackVisualization` panel from the panel `+` menu. Pick a scene,
scrub the timeline, toggle between base- and world-frame views.

## Cuboid rotation convention

`fo.Detection.rotation` is `[rx, ry, rz]` — intrinsic rotation about
each axis. The plugin reads `rot[2]` as yaw (under XYZ Euler), and
the trajectory builder composes per-frame world rotations from the
full `world_from_base` quaternion. Source datasets MUST decompose
the source quaternion with `as_euler("XYZ")` (not `"zyx"`) — the
loader scripts in `fiftyone-tracking-loaders` do this correctly.

## Layout

```
fiftyone-object-tracking-plugin/
├── fiftyone.yml          # plugin manifest (@roboav8r/fiftyone-object-tracking-toolkit)
├── __init__.py           # 4 operator classes + register()
├── _records.py           # TrajectoryRecord + build_track_records + parquet writer
├── _math.py              # SE(3) / quat helpers, gap stats, step velocities, …
├── _schema.py            # SAMPLE_SCHEMA + SIDEBAR_GROUPS + helpers
├── _palette.py           # class → hex color
├── dist/index.umd.js     # BEV panel + TrajectoryRenderer (hand-written UMD)
├── environment.yml       # dev conda env
├── requirements.txt      # runtime python deps
├── install.sh            # local-dev symlink helper
├── README.md
└── LICENSE
```

## Roadmap

Slots reserved for future tracking-specific operators in this plugin:

- `find_similar_trajectories` — DTW-based nearest-neighbor search
  against a reference trajectory.
- `cluster_trajectories` — hierarchical clustering + dendrogram
  threshold panel.
- `flag_qc_outliers` — auto-tag trajectories above configurable
  thresholds on the QC fields.
- `evaluate_predicted_tracks` — GT-vs-predicted track evaluation
  (nuScenes-style; the trajectory parquet schema is already a strict
  superset of nuScenes `TrackingBox`).
