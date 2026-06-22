# fiftyone-object-tracking

FiftyOne plugin for working with grouped 3D tracking datasets:

- **Build per-trajectory datasets** from a tracking source via the
  `build_trajectories` operator. Each trajectory sample is a single
  PNG (BEV plot rendered server-side with matplotlib at build time:
  forward up, left to the left, start `o`, end `x`, ego rectangle at
  origin) plus ~50 filter-friendly scalar facets grouped under
  `Identity` / `Coverage` / `Position (base)` / `Position (world)` /
  `Motion` / `Shape` / `QC`. FO's built-in image renderer handles
  the grid + modal — no custom JS sample renderer.
- **`BEVTrackVisualization` panel** — per-scene bird's-eye-view of
  object trajectories with a timeline scrubber, base ↔ world toggle,
  per-instance presence rows, and pan / zoom.

Source datasets are produced by a separate, private dataloaders repo
(one loader per source format, each emitting the canonical
per-(scene, frame, sensor) grouped schema). This plugin's
trajectory operator is dataset-agnostic — it consumes the canonical
schema, regardless of which loader produced it.

## Install

Into a FiftyOne Enterprise deployment:

```bash
fiftyone plugins download \
    https://github.com/roboav8r/fiftyone-object-tracking
```

For local development, symlink the repo into your plugins dir:

```bash
git clone https://github.com/roboav8r/fiftyone-object-tracking
cd fiftyone-object-tracking
./install.sh   # symlinks into ~/fiftyone/__plugins__/
```

### Python runtime dependencies

The plugin's operators use `numpy`, `scipy`, and `matplotlib` — all
already shipped with the FOE base image. No additional `pip install`
step on the deployment.

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
    @roboav8r/fiftyone-object-tracking/build_trajectories \
    --params '{"source": "<source-tracking>",
               "target": "<target-trajectories>",
               "trajectory_root": "gs://your-bucket/derived/trajectories",
               "overwrite": true}'
```

```python
import fiftyone.operators as foo
op = foo.get_operator("@roboav8r/fiftyone-object-tracking/build_trajectories")
op({
    "source": "delivery-robot-tracking",
    "target": "delivery-robot-trajectories",
    "trajectory_root": "gs://your-bucket/derived/trajectories",
    "overwrite": True,
})
```

### Browse trajectories in the App

After build, open the trajectories dataset. Each cell is a static
PNG (rendered server-side by matplotlib at build time, served by
FO's built-in image renderer) showing the trajectory in BEV —
forward up, left to the left, `o` at the trajectory start, `x` at
the end, faint ego rectangle at origin. The sidebar exposes ~50
filter facets in 7 groups, including the **QC** group:

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
fiftyone-object-tracking/
├── fiftyone.yml          # plugin manifest (@roboav8r/fiftyone-object-tracking)
├── __init__.py           # operator classes + register()
├── _records.py           # TrajectoryRecord + build_track_records
├── _math.py              # SE(3) / quat helpers, gap stats, step velocities, …
├── _dtw.py               # Dynamic Time Warping distance + pairwise matrix
├── _clustering.py        # hierarchical clustering + dendrogram geometry
├── _palette.py           # class → hex color
├── dist/index.umd.js     # Scene / Trajectories / Clusters panel (hand-written UMD)
├── environment.yml       # dev conda env
├── requirements.txt      # runtime python deps (numpy / scipy / matplotlib)
├── install.sh            # local-dev symlink helper
├── README.md
└── LICENSE
```

## Roadmap

Implemented:

- `cluster_trajectories` — DTW + hierarchical clustering of trajectory
  shapes, surfaced as the **Clusters** tab's interactive dendrogram
  (drag-to-cut threshold, click-a-cluster to select).

Slots reserved for future tracking-specific operators in this plugin:

- `find_similar_trajectories` — DTW-based nearest-neighbor search
  against a reference trajectory (reuses `_dtw.py`).
- `flag_qc_outliers` — auto-tag trajectories above configurable
  thresholds on the QC fields.
- `evaluate_predicted_tracks` — GT-vs-predicted track evaluation
  (nuScenes-style).
