# Changelog

All notable changes to this plugin are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this plugin adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it leaves `0.x`.

## [Unreleased]

### Changed

- **The Trajectories grid shows all built scenes by default.** Previously it
  was scoped to the Scene tab's single selected scene, so "Build all scenes"
  built every scene but the grid only ever showed one. The tab now has its own
  **Scene** dropdown (`All scenes` + each built scene); `Select all` / filter /
  tag / export act on the full set. The flat all-scenes grid is render-capped
  (600 cells) with an explicit "Showing N of M" banner — no silent truncation —
  and each cell shows its scene.
- **Tags write through to every slice's detections, not just lidar.** Tagging a
  trajectory now stamps the label tag on the matching `instance._id` across all
  group slices that carry a `detections` field (lidar cuboids **and** the camera
  2D boxes), so the tag shows wherever the track is viewed.
- **Grid and saved-filters bar refresh automatically.** Mutating operators
  (build, filter, apply/clear/delete saved filter, tag) now use the operator
  completion `callback` (FiftyOne ≥2.18) instead of a fixed poll that started on
  click — so a newly saved filter and the regrid appear without a manual refresh.

### Fixed

- **Clicking a cluster now lands the selection in the Trajectories grid.**
  `select_trajectories` had no `resolve_input`, so on FiftyOne ≥ 2.18 the
  executor dropped its `selection` param and the operator silently *cleared*
  the selection instead of setting it. It now declares its input schema and
  takes a list of `{scene_name, track_idx}` (matching `tag_trajectories` /
  `export_trajectories`).

### Added

- **Selected singleton clusters are now highlighted on the dendrogram.** A
  one-trajectory cluster has no below-threshold link of its own (just a leaf
  stub on an above-threshold link), so selecting it showed nothing. Selecting a
  singleton now draws a colored stub + base dot at its leaf position (up to the
  cut line). Multi-member clusters are left as-is (their colored links already
  read clearly), so the dendrogram stays uncluttered.
- **Normalized cluster preview.** The Clusters-tab side preview has a
  **Normalized | Raw** toggle. *Normalized* (default) applies the same
  `origin_normalize` the clustering uses — every path starts at a common origin
  (marked), aligned by its start→end chord — so straight / left / right turns
  separate visually and match the cluster assignment. *Raw* shows the actual
  frame geography. (Ego in the base frame is a single point; use World or
  Scene-local for ego.)
- **Pool trajectories across scenes ("All scenes").** Clustering "All scenes"
  now builds ONE dendrogram over the chosen classes from every scene (stored
  under `__all__`), so e.g. every run's ego path clusters together, or all cars
  across runs. Single-scene clustering is unchanged. Cluster members are now
  tracked as `(scene, track_idx)` pairs end-to-end.
- **Ego is a clusterable class.** Ego appears in the class picker; with cross-
  scene pooling you can cluster the per-run ego paths against each other.
- **Select multiple clusters.** **ctrl/⌘-click** a cluster swatch to add it to
  the selection (plain click still replaces); tag / export act on the union.
- **Save & recall clustering runs.** "Cluster trajectories…" has a **Save run
  as** field; the Clusters tab has a **Saved run** dropdown with **Apply**
  (re-runs that configuration — scene, classes, frame, cut, …) and delete.
  Saved per user; survives rebuilds (it stores the configuration, not the
  cached result).
- **Tag & export a cluster in place.** The Clusters tab now has an inline
  selection toolbar (shown once a cluster is clicked): **Add tag** / **Remove
  tag** (written through to the underlying detection labels), **Export
  (.json)**, and **Clear** — no more hopping to the Trajectories tab.
- **Cluster a subset of classes.** `cluster_trajectories` takes an optional
  multi-select of object classes (default: all), so you can cluster, say, only
  vehicles or only pedestrians — fewer, clearer clusters and faster compute.
  (Ego is excluded — one track per scene.) The class set is part of the per-scene
  params fingerprint, so changing it re-clusters rather than serving stale cache.
- **Delegated execution for `cluster_trajectories`.** Large/all-scenes runs can
  be scheduled as a delegated operation instead of blocking the App (immediate
  execution stays available for small scenes). Requires a delegated-operation
  orchestrator on the deployment; the Clusters tab notes that results arrive
  asynchronously — use ↻ to refresh.
- **Cluster trajectories by shape (DTW + hierarchical clustering).** A new
  **Clusters** tab groups a scene's object trajectories by *shape* using
  Dynamic Time Warping (speed/sampling invariant) + agglomerative clustering,
  and renders the merge tree as an interactive dendrogram. **Drag the cut line**
  to re-cluster live (the cut is recomputed client-side from the linkage matrix
  — no server round-trip); **click a cluster** to select its trajectories. A
  side BEV preview colors the paths by cluster. Selection reuses the existing
  `filter_selection` plumbing, so the Trajectories grid highlights the same
  set. Clustering is per-scene; rebuilding trajectories invalidates it.
- **`cluster_trajectories`** (listed) — compute the DTW distance matrix +
  hierarchical clustering for one scene or all built scenes and store the
  linkage, cut, and dendrogram geometry under `clusters:{scene}`. Shape
  normalization (`origin_normalize`) is on by default so clustering groups by
  shape regardless of position/heading. DTW is `O(N²·T²)`, so paths are
  arc-length-downsampled to `resample_points` (default 30) before DTW and each
  scene is capped at `max_tracks` (default 400, longest tracks kept) with a
  "Clustered N of M" banner — no silent truncation. An optional Sakoe-Chiba
  `band` bounds the warp. (DTW runs serially: spawning subprocesses inside an
  operator worker is unreliable, and resampling + the cap keep it fast.)
- **`get_clusters`** (unlisted) — cheap read path for the Clusters tab, so
  switching scenes never re-runs the `O(N²)` compute.
- **`select_trajectories`** (unlisted) — write a raw `{scene: [track_idx]}`
  selection into `filter_selection`; the seam that converges any selection
  source (a clicked cluster, future similarity search) onto the existing
  highlight path.
- **Tag & export selected trajectories.** The Trajectories grid is now
  multi-selectable: **ctrl/⌘-click** toggles one trajectory, **shift-click**
  range-selects from the last-clicked anchor, and a **Select all** button
  selects every shown object trajectory. A plain click still opens that
  track's patches. A selection toolbar shows the count and exposes the new
  actions.
- **`tag_trajectories`** (unlisted) — add/remove free-form tags on the
  selected trajectories. Tags are **written through to the underlying
  detection label tags** (every `Detection` sharing the trajectory's
  `instance._id` on the lidar slice), so they are durable, filterable in the
  App sidebar, and re-hydrated onto the tracklets on the next
  `build_trajectories`. The in-store tracklets are updated in place so the
  grid reflects tags immediately (shown as chips on each cell).
- **`export_trajectories`** (unlisted) — download the selected trajectories
  as a JSON file (browser download via a base64 `data:` URL, so it works on
  remote deployments). The document is `{scene_name: [record, …]}` where each
  record carries identifiers (`scene_name`, `tracking_id`, `instance_id`,
  `tracking_name`, `track_idx`), the scalar metadata, the `tags`, and the
  per-frame `xy_base`/`xy_world` paths. Intended for an offline pass that
  flags specific `tracking_id`/`instance_id`s for re-annotation.

## [0.3.0] — 2026-06-11

### Changed

- **The panel is now a generic `ObjectTracking` panel with two tabs.**
  Renamed from `BEVTrackVisualization` (breaks references to the old panel
  name in saved workspaces). The **Scene** tab is the previous per-scene
  BEV visualization; the new **Trajectories** tab builds, filters, and
  browses trajectories in-panel.
- **Panel buttons recolored to Voxel51 brand orange** (`#FF6D04`),
  replacing the placeholder blues on the toolbar buttons, the active-tab
  underline, and the "View patches" button. The destructive delete-saved
  (✕) button stays red.
- **`filter_trajectories` is a dynamic, field-typed condition builder.** You
  add conditions one at a time; once you pick a Field, its Value widget is
  typed to that field — a dropdown of **that field's** real values for
  categorical fields (so `Class` offers Cyclist/Car…, `Kind` offers
  object/ego), a number box for numerics, and Yes/No for booleans. A blank
  row appears to add the next condition. Fields show friendly labels
  (`tracking_name`→"Class", `kind`→"Kind (object/ego)"). The stored condition
  spec is unchanged, so saved filters still replay.
- **Saved filters now apply on an explicit "Apply filter" button** rather
  than auto-running when picked from the dropdown.
- **Modal view infers its scene from the open sample.** On the modal
  surface the scene dropdown is hidden; the panel resolves the open sample's
  scene via a `resolve_scene_for_sample` server lookup (matching `_id` or
  `group._id` across slices, so it works whether the modal is showing the
  lidar or a camera slice) and auto-selects it (re-inferring as you navigate).
- **Trajectories are now ephemeral — no trajectories dataset is created.**
  `build_trajectories` no longer materializes a per-trajectory FiftyOne
  dataset; it extracts tracklets (scalars + per-frame XY) into a
  dataset-scoped `ExecutionStore`. Removed `_schema.py`, `_thumbnail.py`,
  and the `matplotlib` dependency.

### Added

- **Trajectories tab.** In-panel grid of client-side mini-BEV thumbnails;
  clicking a cell opens that track's patches via `view_track_patches`.
- **`get_trajectories`** (unlisted) — reads built tracklets + the current
  filter selection for the grid.
- **`filter_trajectories`** — select tracklets by one or more field/op/value
  conditions (numeric/categorical/boolean) combined with AND/OR; writes a
  `{scene: [track_idx]}` selection for QC or scene/data mining.
- **Per-user saveable filters** — `filter_trajectories` can `save_as` a
  named spec in a global store keyed by user; `list_trajectory_filters` /
  `delete_trajectory_filter` back the tab's saved-filters dropdown.
- **Ego / World frame toggle on the Trajectories grid.** Mini-BEV
  thumbnails can plot either the ego-relative (base-frame) path or the
  absolute world-frame path; in world frame the ego-origin marker and the
  force-include-origin framing are dropped (the origin isn't meaningful).
- **`clear_trajectory_filter`** (unlisted) — clears the active filter
  selection with no required inputs (see Fixed).
- **Scrubber syncs to FiftyOne's native modal timeline.** Instead of custom
  playback controls, the Scene-tab scrubber follows the modal's native
  timeline (`@fiftyone/playback`): native play/loop/speed advance the scrubber
  and BEV markers, and dragging the scrubber seeks the native looker while
  playback continues. Falls back to `open_sample` seeking when no native
  timeline is present.

### Fixed

- **Modal-surface detection fixed (`@fiftyone/spaces` PanelContext scope).**
  The panel relied on an `isModalPanel` prop that FiftyOne no longer passes to
  plugin components, so every modal-only behavior silently no-op'd on the modal
  surface (scene dropdown wasn't hidden, scene wasn't inferred, the scrubber
  didn't drive the looker). It now reads `usePanelContext().scope === "modal"`,
  which re-activates scene inference, the dropdown hide, and timeline sync.

- **"Clear filter" no longer throws "Failed to execute an operation."**
  The button executed `filter_trajectories` with no `combinator`, which
  failed input validation outside the operator prompt. Clearing now uses a
  dedicated `clear_trajectory_filter` operator that just deletes the
  selection; `delete_trajectory_filter` also tolerates a missing key.
- **Scene scrubber drives the modal looker.** When no native timeline is
  present, scrubbing in **modal** view jumps the open looker to the scrubbed
  frame's grouped sample via the built-in `open_sample` operator (group-slice
  aware through `useSetExpandedSample`); with a native timeline it seeks that
  instead (see Added). Previously this never ran because modal-surface
  detection was broken (see above).

- **BEV panel camera-mirror thumbnail now renders local media.** The
  inline thumbnail loaded cloud (gs:///s3://) frames but showed a broken
  image for local-filesystem datasets: `get_camera_frame_urls` returned a
  raw local path that the browser can't load. The frontend now resolves
  every thumbnail URL through `fos.getSampleSrc`, which passes signed/HTTP
  URLs through unchanged and routes local paths through the App's `/media`
  server. The operator now classifies paths explicitly by file system
  (HTTP/local pass through; cloud is signed) instead of relying on a broad
  `except` around `fos.get_url`.

## [0.2.0] — 2026-05-19

Initial public release. Renames the plugin to its shipping namespace
(`@roboav8r/fiftyone-object-tracking`) and replaces the parquet
payload + custom JS sample renderer with server-rendered PNG
thumbnails (matplotlib), so the plugin installs into any FOE
deployment with no extra `pip install` step.

### Added

- **`build_trajectories` operator** — consumes a grouped tracking
  dataset (lidar slice with 3D cuboid detections, `tracking_id` +
  `segment_index` stamped on each detection, `world_to_base` SE(3)
  per frame) and emits a sibling per-trajectory FiftyOne dataset.
  Each trajectory becomes one FO sample whose `filepath` is a
  ~512 × 512 px PNG (BEV plot rendered server-side by matplotlib at
  build time) and whose sample-level fields are ~50 filter-friendly
  scalars in 8 sidebar groups (`tags`, `Identity`, `Coverage`,
  `Position (base)`, `Position (world)`, `Motion`, `Shape`, `QC`,
  plus `metadata`).
- **`BEVTrackVisualization` panel** — per-scene bird's-eye-view of
  object trajectories with a timeline scrubber, base ↔ world toggle,
  per-instance presence rows, and pan / zoom.
- **`list_tracking_scenes` + `get_scene_track_payload`** — unlisted
  utility operators the BEV panel calls into.
- **QC sidebar group** on the trajectories dataset surfaces
  annotation-quality signals without splitting tracks:
  `n_distinct_classes`, `tracking_names_distinct`, `max_step_jump_m`,
  `max_gap_s`.

### Plot convention

- Image-up = `+x_base` (robot forward), image-left = `+y_base`
  (robot left).
- Each trajectory: per-fragment solid polylines + dashed bridges
  across gaps. Start marked with `o`, end with `x`, in the
  trajectory's class color.
- Object trajectories include the ego rectangle at origin sized
  from the source dataset's `info["ego_size_lwh_m"]` (falling back
  to the OSDaR23 rail constant `[24.0, 2.9, 4.0]` if unset).
- Ego trajectories plot `(x_scene_local, y_scene_local)` instead so
  the path isn't a degenerate dot at the base-frame origin.

### Runtime dependencies

`numpy`, `scipy`, `matplotlib` — all already in the FiftyOne
Enterprise base image. No additional install step on the deployment.

### Requirements

- FiftyOne Enterprise ≥ 2.18.0.
- Admin permissions on the deployment to upload the plugin.

[0.3.0]: https://github.com/roboav8r/fiftyone-object-tracking/releases/tag/v0.3.0
[0.2.0]: https://github.com/roboav8r/fiftyone-object-tracking/releases/tag/v0.2.0
