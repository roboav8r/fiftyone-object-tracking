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

### Python runtime dependencies

The plugin's operators need `numpy`, `scipy`, `pandas`, and `pyarrow`
(see `requirements.txt`). `numpy` / `scipy` / `pandas` are already in
the FOE base image; **`pyarrow` is not** and must be installed into
every service that executes plugin operator code.

If `pyarrow` is missing, the `read_trajectory_payload` operator will
fail with `ModuleNotFoundError("No module named 'pyarrow'")` and the
`TrajectoryRenderer` grid cells will show that error instead of the
BEV plot.

#### Which services need pyarrow?

FiftyOne Teams runs plugin Python in **multiple** services:

| Service | Role | Needs `pyarrow`? |
|---|---|---|
| `fiftyone-app` | App-server-side operator execution for non-Teams installs | yes |
| `teams-plugins` | Synchronous plugin-operator execution (Teams) | **yes** (this is where `read_trajectory_payload` runs in your deployment) |
| `teams-do` (× N replicas) | Delegated-operator workers | yes (if you'll ever run `build_trajectories` in delegated mode) |
| `teams-api` | Teams API server | optional today, but recommended for future-proofing |
| `teams-app` | React UI image; no Python operators | no |
| `teams-cas` | Auth service | no |

The exact service names depend on your compose project / helm release.

#### Quick check (any deployment)

```bash
# Run against each FOE Python service to confirm pyarrow is reachable
docker exec <service-container> python -c "import pyarrow; print(pyarrow.__version__)"
# (or, in a helm pod:)
kubectl exec -n <namespace> <pod> -- python -c "import pyarrow; print(pyarrow.__version__)"
```

#### Quick fix (short-lived)

Drop pyarrow into running containers / pods. **Reverted on next
container restart**, so this is for testing only:

```bash
# Docker Compose
for c in <project>-fiftyone-app-1 <project>-teams-plugins-1 \
         <project>-teams-do-1 <project>-teams-do-2 <project>-teams-do-3 \
         <project>-teams-api-1; do
    docker exec "$c" pip install --no-cache-dir pyarrow
done

# Helm / Kubernetes
for pod in $(kubectl get pods -n <ns> -l app.kubernetes.io/name=fiftyone -o name); do
    kubectl exec -n <ns> "$pod" -- pip install --no-cache-dir pyarrow
done
```

#### Durable fix — bake into a custom image (recommended)

Build an image FROM the FOE base, add `pyarrow`, and point the
operator-running services at that image. Survives container
recreation, helm upgrades, etc.

**Dockerfile** (next to your compose.yaml):

```dockerfile
# Dockerfile.fiftyone-with-pyarrow
ARG FIFTYONE_VERSION
FROM voxel51/fiftyone-app:${FIFTYONE_VERSION}
RUN pip install --no-cache-dir pyarrow
```

**Docker Compose** — override the image for every Python-running service:

```yaml
# compose.override.yaml
services:
  fiftyone-app: &with-pyarrow
    build:
      context: .
      dockerfile: Dockerfile.fiftyone-with-pyarrow
      args:
        FIFTYONE_VERSION: "2.18.0"   # match your deployment
    image: local/fiftyone-app-with-pyarrow:2.18.0
  teams-plugins:
    <<: *with-pyarrow
  teams-do:
    <<: *with-pyarrow

# then
docker compose build
docker compose up -d
```

**Helm** — push the custom image to a registry the cluster can reach,
then override the image in `values.yaml` for the relevant services
(the exact key names depend on your chart version; check `helm show
values voxel51/fiftyone-teams-app`):

```yaml
# values.yaml
apiSettings:
  image:
    repository: my-registry/fiftyone-app-with-pyarrow
    tag: 2.18.0
pluginsSettings:                  # for Teams; key sometimes named teamsPlugins
  image:
    repository: my-registry/fiftyone-app-with-pyarrow
    tag: 2.18.0
delegatedOperatorExecutorSettings: # key sometimes teamsDo / delegatedOperator
  image:
    repository: my-registry/fiftyone-app-with-pyarrow
    tag: 2.18.0
appSettings:
  image:
    repository: my-registry/fiftyone-app-with-pyarrow
    tag: 2.18.0

# then
helm upgrade --install fiftyone voxel51/fiftyone-teams-app -f values.yaml
```

If your chart exposes a `extraEnvFrom` or `initContainers` hook
instead, an init container that runs `pip install pyarrow` against a
shared volume can work but is fiddlier than rebuilding the image.

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
