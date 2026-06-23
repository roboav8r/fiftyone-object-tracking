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
- **`ObjectTracking` panel** — per-scene bird's-eye-view of
  object trajectories with a timeline scrubber, base ↔ world toggle,
  per-instance presence rows, and pan / zoom.

Source datasets are produced by a separate, private dataloaders repo
(one loader per source format, each emitting the canonical
per-(scene, frame, sensor) grouped schema). This plugin's
trajectory operator is dataset-agnostic — it consumes the canonical
schema, regardless of which loader produced it.

## Requirements

<!-- TODO -->

## Highlights

<!-- TODO -->

## Install

The plugin installs like any other FiftyOne plugin — see the canonical
docs for [FiftyOne (OSS)](https://docs.voxel51.com/plugins/using_plugins.html)
and [FiftyOne Enterprise](https://docs.voxel51.com/enterprise/plugins.html).

**FiftyOne (OSS):**

```bash
fiftyone plugins download https://github.com/roboav8r/fiftyone-object-tracking
```

**FiftyOne Enterprise:** upload the plugin from the **Plugins** page in
the Enterprise UI (drag-and-drop a ZIP), or via the Management SDK:

```python
import fiftyone.management as fom
fom.upload_plugin("/path/to/fiftyone-object-tracking", overwrite=True)
```

## Documentation

| Doc | What it covers |
|---|---|
| [docs/DATASET_SCHEMA.md](docs/DATASET_SCHEMA.md) | Canonical source-dataset schema + cuboid rotation convention |
| [docs/SCENE.md](docs/SCENE.md) | Per-scene BEV panel (timeline scrubber, base ↔ world) |
| [docs/TRAJECTORIES.md](docs/TRAJECTORIES.md) | Build + browse per-trajectory datasets |
| [docs/CLUSTERS.md](docs/CLUSTERS.md) | DTW + hierarchical clustering of trajectory shapes |

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
├── docs/                 # SCENE / TRAJECTORIES / CLUSTERS / DATASET_SCHEMA
├── environment.yml       # dev conda env
├── requirements.txt      # runtime python deps (numpy / scipy / matplotlib)
├── install.sh            # local-dev symlink helper
├── README.md
└── LICENSE
```

## Future Work

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
```
