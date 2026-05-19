# Changelog

All notable changes to this plugin are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this plugin adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it leaves `0.x`.

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

[0.2.0]: https://github.com/roboav8r/fiftyone-object-tracking/releases/tag/v0.2.0
