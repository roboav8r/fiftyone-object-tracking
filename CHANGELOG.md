# Changelog

All notable changes to this plugin are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this plugin adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it leaves `0.x`.

## [Unreleased]

### Added

- **`split_track` operator** (listed) — splits one selected track at the
  scrubber's current frame: detections with `frame_idx < split_frame`
  keep the original `instance._id`; detections at or after the split
  frame are reassigned to a freshly minted `fo.Instance()`. The mutation
  walks every group slice (lidar + cameras) via
  `select_group_slices(_allow_mixed=True)` so the new instance id is
  consistent across slices.
- **`join_tracks` operator** (listed) — merges 2+ selected tracks onto
  a single `fo.Instance` (the one whose earliest detection has the
  lowest `frame_idx` wins). Cross-slice, same mutation pattern as split.
- **BEV panel: detections-field selector** — header `Field:` input
  controls both the field the panel *visualizes* and the field
  Split/Join write into. Defaults to `"detections"`; switch to any
  custom name (e.g. `"corrected_detections"`) for non-destructive edits.
  Split/Join copy `detections` into the target field on first edit if
  the target doesn't yet exist.
- **BEV panel: Split / Join buttons** — Split is enabled when exactly
  one track is selected and the scrubber has a position; Join is enabled
  when ≥2 tracks are selected. Post-edit the panel refetches the payload
  for the active field and re-selects the resulting instance.
- **Multi-surface operator invocation** — both operators also work
  from the operator palette, against `ctx.selected_samples` (grid
  selection), or against `ctx.selected_labels` (embeddings-panel lasso).
  When no `split_frame` is provided, the operator derives it from
  `min(frame_idx)` across the selected samples / labels.

### Changed

- **BEV panel header reorganized into two rows** sandwiching the chart:
  a top *inspection* row (scene / view field / coord / counts / preview
  camera) and a new edit-controls row beneath the chart (View patches /
  Edit field / Split / Merge). The single-field model from the prior
  iteration is replaced by independent `viewField` (chart) and
  `editField` (Split/Join target); after a successful edit, the chart
  auto-flips to the edited field so the result is visible. Coordinate
  toggle relabelled "View:" → "Coord:". Default edit-field name is
  `detections_corrected`.
- **`split_track`** now declares `split_frame` as `required=True`. The
  BEV panel button still passes it explicitly from the scrubber (so its
  UX is unchanged), but invocations from the operator palette, grid
  sample selection, or embeddings-panel lasso now surface an input form
  asking for the split frame rather than silently inferring it from the
  selection. `_resolve_track_edit_targets` no longer returns a derived
  frame.
- **`get_scene_track_payload`** now takes an optional `source_field`
  param (default `"detections"`) and reads detections from that field,
  echoing it back as `source_field` in the result so the panel can key
  its cache by `(scene, field)`.
- **`build_track_records`** (`_records.py`) takes a kw-only `field_path`
  (default `"detections"`) — the six hardcoded `detections.detections.*`
  paths now derive from this prefix.

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
