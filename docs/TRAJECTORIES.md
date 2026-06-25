# Trajectories

The trajectory workflow builds a per-trajectory dataset from a tracking
source. Each trajectory sample is a single PNG (BEV plot rendered
server-side with matplotlib at build time: forward up, left to the left,
start `o`, end `x`, ego rectangle at origin) plus ~50 filter-friendly
scalar facets grouped under `Identity` / `Coverage` / `Position (base)` /
`Position (world)` / `Motion` / `Shape` / `QC`. FO's built-in image
renderer handles the grid + modal — no custom JS sample renderer.

> Source dataset schema: see [DATASET_SCHEMA.md](DATASET_SCHEMA.md).

## Build trajectories from the App

Open the source tracking dataset → operator palette → "Build
trajectories dataset" → fill the form (target name, trajectory_root,
overwrite) → run. The new trajectories dataset appears in the App's
dataset selector.

- **Build per-trajectory datasets** from a tracking source via the
  `build_trajectories` operator. Each trajectory sample is a single
  PNG (BEV plot rendered server-side with matplotlib at build time:
  forward up, left to the left, start `o`, end `x`, ego rectangle at
  origin) plus ~50 filter-friendly scalar facets grouped under
  `Identity` / `Coverage` / `Position (base)` / `Position (world)` /
  `Motion` / `Shape` / `QC`. FO's built-in image renderer handles
  the grid + modal — no custom JS sample renderer.

## Build trajectories from the SDK / CLI

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

## Browse trajectories in the App

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
