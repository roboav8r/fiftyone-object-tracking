# fiftyone-object-tracking

FiftyOne Object Tracking (FOOT) is a [FiftyOne Plugin](https://docs.voxel51.com/plugins/index.html) designed to simplify the development of multi-object tracking (MOT) systems. 

As robots and intelligent systems operate in dynamic, real-world environments, it's important for them to perceive nearby objects and predict their future states in real-time. MOT provides the classes and states of nearby objects, making it a solid foundation for a physical AI perception stack. MOT is used in robotics, autonomous vehicles, security/defense, smart spaces, retail applications, and many other areas.

Despite their importance to physical AI, development and debugging MOT systems is notoriously complex: there are multiple models and parameters that change depending on the operating environment and object being tracked.

To this end, FOOT aims to make the MOT development process more intuitive, in true FiftyOne fashion. Foot can be used for:
- Visualizing tracking scenes and sequences
- Inspecting and quality checking track annotations
- Retrieving similar tracks and trajectories of known objects
- (Planned) Training, applying, and evaluating MOT models to datasets

**If you are a developer or user of multiple object tracking, please read on!**

## Highlights

### [Tracking scene visualization](docs/SCENE.md)

Visualize all objects and the ego-vehicle in a scene via a birds-eye-view (BEV) panel. Visualize scene progression in vehicle or world frame, and seek specific moments using the timeline scrubber.

For visualizing data streams and replaying specific episodes in your tracking dataset to gain a high-level overview.

### [Trajectory panel](docs/TRAJECTORIES.md)
Extract, inspect, and quality check trajectories for each 
object in the dataset. View the detections for a specific object. 
Tag and export for further usage.

Useful for identifying track annotation errors or quality checking.

### [Clustering](docs/CLUSTERS.md)
Group similar-shaped trajectories using dynamic time warping 
- hierarchical clustering (DTW-HC). Then, tag and export them.

Useful for finding similar categories of trajectories 
(e.g. vehicle left turns, pedestrian crossingss) and the
parent scenes.

## Requirements

- FiftyOne installed (open-source or FiftyOne Enterprise)
- A FiftyOne dataset that adheres to the [docs/DATASET_SCHEMA.md](dataset schema)

Source datasets are produced by a separate dataloader (not included).
which emits a canonical per-(scene, frame, sensor) grouped schema. 
This plugin is dataset-agnostic; it consumes the canonical
schema, regardless of how it was produced or the domain.


## Installation

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
## Quickstart Guide
[TODO]

## Detailed Usage
For detailed usage instructions, refer to the appropriate documentation pages for the task you'd like to perform:

| Doc | What it covers |
|---|---|
| [docs/DATASET_SCHEMA.md](docs/DATASET_SCHEMA.md) | Canonical source-dataset schema + conventions |
| [docs/SCENE.md](docs/SCENE.md) | Per-scene BEV panel (timeline scrubber, base ↔ world) |
| [docs/TRAJECTORIES.md](docs/TRAJECTORIES.md) | Build + browse per-trajectory datasets |
| [docs/CLUSTERS.md](docs/CLUSTERS.md) | DTW + hierarchical clustering of trajectory shapes |

## Repo Organization

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
