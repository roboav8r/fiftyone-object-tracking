"""Operators for the FiftyOne object-tracking toolkit plugin.

Three operators, exposed through ``register()``:

  - ``list_tracking_scenes``    (unlisted) — enumerate scenes +
                                group-slice names for the BEV panel
  - ``get_scene_track_payload`` (unlisted) — one-shot per-scene
                                trajectory bundle for the BEV panel
  - ``build_trajectories``      (listed)   — consume a grouped
                                tracking dataset and emit a sibling
                                per-trajectory dataset, with each
                                trajectory rendered as a single BEV
                                PNG by matplotlib. FO's built-in
                                image renderer handles the grid +
                                modal; no custom JS sample renderer
                                and no PyArrow / Parquet roundtrip
                                in the loop.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Any

import numpy as np
from bson import ObjectId

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F

from ._records import (
    DEFAULT_EGO_SIZE_LWH,
    TrajectoryRecord,
    build_track_records,
)
from ._schema import (
    SAMPLE_SCHEMA,
    declare_schema,
    set_sidebar_groups,
)
from ._thumbnail import render_trajectory_thumbnail


DEFAULT_LIDAR_SLICE = "lidar"
DEFAULT_SOURCE_FIELD = "detections"
DEFAULT_TARGET_FIELD = "detections_corrected"


# -----------------------------------------------------------------------------
# Shared helpers (used only by the JS-facing utility operators below)
# -----------------------------------------------------------------------------

def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw about world up (z) from a unit quaternion (xyzw)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _footprint_corners(cx: float, cy: float, yaw: float,
                       length: float, width: float):
    """4 (x, y) top-down corners for a rotated rectangle of size L × W."""
    c, s = math.cos(yaw), math.sin(yaw)
    hL, hW = length / 2.0, width / 2.0
    local = ((+hL, +hW), (+hL, -hW), (-hL, -hW), (-hL, +hW))
    return [(cx + c * lx - s * ly, cy + s * lx + c * ly) for lx, ly in local]


def _resolve_lidar_slice(dataset) -> str | None:
    """First slice named exactly 'lidar', else first slice with 'lidar'
    in its name. Mirrors the existing logic in ListTrackingScenes.
    """
    group_slices = list(dataset.group_slices or [])
    if DEFAULT_LIDAR_SLICE in group_slices:
        return DEFAULT_LIDAR_SLICE
    return next((s for s in group_slices if "lidar" in s.lower()), None)


def _ensure_target_field(dataset, source_field: str, target_field: str) -> bool:
    """If target_field is missing from the schema, copy source_field into
    it across every group slice via ``clone_sample_field``. Idempotent.
    Raises ValueError if neither field exists. Returns True if a copy
    actually happened.
    """
    if source_field == target_field:
        return False
    if dataset.get_field(target_field) is not None:
        return False
    if dataset.get_field(source_field) is None:
        raise ValueError(
            f"Cannot copy: source field {source_field!r} not present on "
            f"dataset {dataset.name!r}"
        )
    # clone_sample_field adds the field to the schema AND deep-copies every
    # sample's value across all group slices. set_field+save would require
    # target_field to already exist in the schema, which is the bug this
    # replaces.
    dataset.clone_sample_field(source_field, target_field)
    return True


def _instances_from_selected_samples(dataset, sample_ids, target_field):
    """Look up distinct instance._id hexes plus the (min frame_idx, set
    of scene_names) carried by the given sample ids in target_field.
    """
    if not sample_ids:
        return [], None, set()
    view = dataset.select(sample_ids)
    scenes, frames, inst_lists = view.values([
        "scene_name", "frame_idx",
        f"{target_field}.detections.instance._id",
    ])
    hexes: set[str] = set()
    for inst_list in inst_lists or []:
        if not inst_list:
            continue
        for oid in inst_list:
            if oid is not None:
                hexes.add(str(oid))
    min_frame = min((int(f) for f in (frames or []) if f is not None), default=None)
    scene_set = {s for s in (scenes or []) if s}
    return sorted(hexes), min_frame, scene_set


def _reassign_instances_across_slices(
    dataset, *, target_field, scene_name,
    match_instance_oids, replace_when_hex_in,
    new_instance, extra_match=None, log_prefix="reassign",
):
    """Walk every group slice, find detections whose ``instance._id`` is
    in ``match_instance_oids`` (and optionally satisfy ``extra_match``),
    and replace ``det.instance`` with ``new_instance``.

    Per-slice iteration (not ``_allow_mixed=True``) so
    ``iter_samples(autosave=True)`` reliably persists writes back. Returns
    ``(n_detections, n_samples)``.
    """
    n_det = 0
    n_samp = 0
    per_slice = {}
    slices = list(dataset.group_slices or [])
    if not slices:
        slices = [None]  # non-grouped fallback (shouldn't happen in this plugin)

    # Use Mongo dict form for the instance filter. F(path).is_in([oid])
    # on a nested-array-of-subdocs path doesn't generate the elemMatch
    # we want — it compares the whole array as a single value and matches
    # nothing. The dict form maps directly to MongoDB's
    # `{array_path: {$in: [...]}}` which IS elemMatch for array fields.
    inst_filter = {
        f"{target_field}.detections.instance._id": {"$in": list(match_instance_oids)},
    }

    for slice_name in slices:
        view = (
            dataset.select_group_slices(slice_name)
            if slice_name is not None else dataset.view()
        )
        if scene_name:
            view = view.match(F("scene_name") == scene_name)
        n_scene = view.count() if scene_name else None
        view = view.match(inst_filter)
        n_after_inst = view.count()
        if extra_match is not None:
            view = view.match(extra_match)
        n_after_extra = view.count()

        slice_dets = 0
        slice_samps = 0
        for sample in view.iter_samples(autosave=True):
            dets = sample[target_field]
            if dets is None:
                continue
            changed = False
            for det in dets.detections:
                if det.instance is None:
                    continue
                if det.instance.id in replace_when_hex_in:
                    det.instance = new_instance
                    slice_dets += 1
                    changed = True
            if changed:
                sample[target_field] = dets
                slice_samps += 1
        per_slice[slice_name or "_default"] = {
            "n_scene": n_scene,
            "n_after_inst_filter": n_after_inst,
            "n_after_extra_filter": n_after_extra,
            "n_det_changed": slice_dets,
            "n_samp_touched": slice_samps,
        }
        n_det += slice_dets
        n_samp += slice_samps

    print(
        f"[{log_prefix}] target_field={target_field!r} scene={scene_name!r} "
        f"matched_hexes={sorted(replace_when_hex_in)!r} "
        f"per_slice={per_slice} totals=(det={n_det}, samp={n_samp})"
    )
    return n_det, n_samp, per_slice


def _instances_from_patches_view(ctx):
    """If ``ctx.view`` is a ``PatchesView`` and patches are selected,
    look them up *through the view* (sample ids in a patches view are
    patch ids, not parent-sample ids — ``dataset.select(patch_ids)``
    silently returns nothing).

    Returns ``(hexes, min_frame, scenes, patches_field)`` or
    ``(None, None, None, None)`` if not a patches view / no selection.
    """
    view = getattr(ctx, "view", None)
    if view is None:
        return None, None, None, None
    # PatchesView (and EvaluationPatchesView) carry the source field
    # name on ``_patches_field``. Generic Views won't have this.
    patches_field = getattr(view, "_patches_field", None)
    if patches_field is None:
        return None, None, None, None
    selected = list(ctx.selected or [])
    if not selected:
        return None, None, None, None
    try:
        sub = view.select(selected)
        # In a patches view, the source list field is flattened: each
        # row is a single Detection, accessed directly as
        # ``{patches_field}`` (not ``{patches_field}.detections``).
        oids = sub.values(f"{patches_field}.instance._id")
        scene_vals = sub.values("scene_name")
        frame_vals = sub.values("frame_idx")
    except Exception:
        return None, None, None, None
    hexes = sorted({str(o) for o in (oids or []) if o})
    scenes = {s for s in (scene_vals or []) if s}
    frames = [int(f) for f in (frame_vals or []) if f is not None]
    min_frame = min(frames) if frames else None
    return hexes, min_frame, scenes, patches_field


def _instances_from_selected_labels(dataset, selected_labels, target_field):
    """Resolve the (instance hex, min frame_idx, scene_names) tuple for
    a list of ctx.selected_labels entries. Each entry has at least
    ``label_id`` and ``sample_id``; ``field`` and ``frame_number`` are
    ignored for grouped sample-level detections (we always read
    ``target_field`` and treat all matching detections as candidates).
    """
    if not selected_labels:
        return [], None, set()
    sample_ids = sorted({sl["sample_id"] for sl in selected_labels})
    label_ids = {sl["label_id"] for sl in selected_labels}

    view = dataset.select(sample_ids)
    scenes, frames, ids_per_sample, oids_per_sample = view.values([
        "scene_name", "frame_idx",
        f"{target_field}.detections.id",
        f"{target_field}.detections.instance._id",
    ])
    hexes: set[str] = set()
    for det_ids, det_oids in zip(ids_per_sample or [], oids_per_sample or []):
        if not det_ids:
            continue
        for det_id, oid in zip(det_ids, det_oids or []):
            if det_id in label_ids and oid is not None:
                hexes.add(str(oid))
    min_frame = min((int(f) for f in (frames or []) if f is not None), default=None)
    scene_set = {s for s in (scenes or []) if s}
    return sorted(hexes), min_frame, scene_set


# -----------------------------------------------------------------------------
# 1. list_tracking_scenes
# -----------------------------------------------------------------------------

class ListTrackingScenes(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="list_tracking_scenes",
            label="List tracking scenes",
            unlisted=True,
        )

    def execute(self, ctx) -> dict[str, Any]:
        group_slices = list(ctx.dataset.group_slices or [])
        lidar_slice = DEFAULT_LIDAR_SLICE if DEFAULT_LIDAR_SLICE in group_slices \
            else next((s for s in group_slices if "lidar" in s.lower()), None)

        if lidar_slice is None:
            return {
                "scenes": [],
                "group_slices": {"all": group_slices, "lidar": None},
                "error": "No lidar group slice found on this dataset.",
            }

        # Always go through ctx.dataset, not ctx.view. The BEV panel
        # works on full scenes regardless of grid filtering, and ctx.view
        # may be a PatchesView / ClipsView / etc. whose
        # select_group_slices() raises ("<class 'PatchesView'> has no
        # groups").
        view = ctx.dataset.select_group_slices(lidar_slice)

        scene_names, frame_idxs, sample_ids = view.values(
            ["scene_name", "frame_idx", "id"]
        )

        scenes: dict[str, dict] = {}
        for scene, fidx, sid in zip(scene_names, frame_idxs, sample_ids):
            if scene is None or fidx is None:
                continue
            entry = scenes.setdefault(
                scene, {"scene_name": scene, "n_frames": 0,
                        "first_frame_idx": fidx,
                        "first_lidar_sample_id": sid}
            )
            entry["n_frames"] += 1
            if fidx < entry["first_frame_idx"]:
                entry["first_frame_idx"] = fidx
                entry["first_lidar_sample_id"] = sid

        # Discover Detections-typed fields on the lidar slice so the BEV
        # panel can populate the "View field" dropdown. Includes the
        # canonical "detections" plus any sibling fields (e.g. user-
        # created "detections_corrected" after a Split/Merge edit).
        det_fields: list[str] = []
        try:
            schema = ctx.dataset.select_group_slices(lidar_slice).get_field_schema(
                ftype=fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Detections,
            )
            det_fields = sorted(schema.keys())
        except Exception:
            det_fields = []

        return {
            "scenes": sorted(scenes.values(), key=lambda s: s["scene_name"]),
            "group_slices": {"all": group_slices, "lidar": lidar_slice},
            "detections_fields": det_fields,
        }


# -----------------------------------------------------------------------------
# 2. get_scene_track_payload
# -----------------------------------------------------------------------------

class GetSceneTrackPayload(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_scene_track_payload",
            label="Get scene track payload",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("scene_name", required=True)
        inputs.str("lidar_slice", default=DEFAULT_LIDAR_SLICE)
        inputs.str("source_field", default=DEFAULT_SOURCE_FIELD)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        scene_name = ctx.params["scene_name"]
        lidar_slice = ctx.params.get("lidar_slice") or DEFAULT_LIDAR_SLICE
        source_field = ctx.params.get("source_field") or DEFAULT_SOURCE_FIELD

        if ctx.dataset.get_field(source_field) is None:
            return {
                "scene_name": scene_name,
                "source_field": source_field,
                "error": (
                    f"Field {source_field!r} does not exist on dataset "
                    f"{ctx.dataset.name!r}. Make an edit (Split/Merge) "
                    f"to auto-create it from 'detections', or type a "
                    f"different View field."
                ),
            }

        lidar_view = ctx.dataset.select_group_slices(lidar_slice).match(
            F("scene_name") == scene_name
        ).sort_by("frame_idx")

        (
            sample_ids,
            frame_idxs,
            m_ts,
            wtb_matrices,
            wtb_translations,
            wtb_quats,
            det_instance_ids,
            det_labels,
            det_tracking_ids,
            det_locations,
            det_rotations,
            det_dimensions,
        ) = lidar_view.values([
            "id",
            "frame_idx",
            "m_frame_timestamp",
            "world_to_base.matrix_4x4_row_major",
            "world_to_base.translation",
            "world_to_base.quaternion_xyzw",
            f"{source_field}.detections.instance._id",
            f"{source_field}.detections.label",
            f"{source_field}.detections.tracking_id",
            f"{source_field}.detections.location",
            f"{source_field}.detections.rotation",
            f"{source_field}.detections.dimensions",
        ])

        n_frames = len(sample_ids)
        if n_frames == 0:
            return {
                "scene_name": scene_name,
                "error": f"No lidar samples found for scene '{scene_name}'.",
            }

        ego_world_x: list[float] = []
        ego_world_y: list[float] = []
        ego_world_yaw: list[float] = []
        for tr, q in zip(wtb_translations, wtb_quats):
            if tr is not None and len(tr) >= 2:
                ego_world_x.append(float(tr[0]))
                ego_world_y.append(float(tr[1]))
            else:
                ego_world_x.append(float("nan"))
                ego_world_y.append(float("nan"))
            if q is not None and len(q) == 4:
                ego_world_yaw.append(_quat_to_yaw(*[float(v) for v in q]))
            else:
                ego_world_yaw.append(float("nan"))

        per_frame_T = []
        for m in wtb_matrices:
            if m is not None and len(m) == 16:
                per_frame_T.append(np.asarray(m, dtype=np.float64).reshape(4, 4))
            else:
                per_frame_T.append(None)

        tracks: dict[str, dict] = {}
        for f_i in range(n_frames):
            fidx = frame_idxs[f_i]
            T = per_frame_T[f_i]

            ids_f = det_instance_ids[f_i] or []
            labels_f = det_labels[f_i] or []
            tids_f = det_tracking_ids[f_i] or []
            locs_f = det_locations[f_i] or []
            rots_f = det_rotations[f_i] or []
            dims_f = det_dimensions[f_i] or []

            for instance_id, label, tracking_id, loc, rot, dims in zip(
                ids_f, labels_f, tids_f, locs_f, rots_f, dims_f
            ):
                if instance_id is None or loc is None or dims is None:
                    continue
                instance_id = str(instance_id)
                bx = float(loc[0])
                by = float(loc[1])
                bz = float(loc[2]) if len(loc) >= 3 else 0.0
                # detection.rotation is [rx, ry, rz]; the BEV panel only
                # uses yaw (= rz under XYZ Euler when pitch/roll ≈ 0).
                yaw_b = float(rot[2]) if rot is not None and len(rot) >= 3 else 0.0
                L = float(dims[0])
                W = float(dims[1])
                base_corners = _footprint_corners(bx, by, yaw_b, L, W)

                if T is not None:
                    p_w = T @ np.array([bx, by, bz, 1.0])
                    wx, wy = float(p_w[0]), float(p_w[1])
                    yaw_world_base = math.atan2(T[1, 0], T[0, 0])
                    yaw_w = yaw_world_base + yaw_b
                    world_corners = _footprint_corners(wx, wy, yaw_w, L, W)
                else:
                    wx = float("nan")
                    wy = float("nan")
                    yaw_w = float("nan")
                    world_corners = [(float("nan"), float("nan"))] * 4

                t = tracks.setdefault(
                    instance_id,
                    {
                        "instance_id": instance_id,
                        "tracking_id": (
                            None if tracking_id is None else str(tracking_id)
                        ),
                        "label": label or "unknown",
                        "frames": [],
                        "base": {"x": [], "y": [], "yaw": [], "L": [], "W": [],
                                 "corners": []},
                        "world": {"x": [], "y": [], "yaw": [], "L": [], "W": [],
                                  "corners": []},
                    },
                )
                t["frames"].append(int(fidx))
                t["base"]["x"].append(bx)
                t["base"]["y"].append(by)
                t["base"]["yaw"].append(yaw_b)
                t["base"]["L"].append(L)
                t["base"]["W"].append(W)
                t["base"]["corners"].append([list(c) for c in base_corners])
                t["world"]["x"].append(wx)
                t["world"]["y"].append(wy)
                t["world"]["yaw"].append(yaw_w)
                t["world"]["L"].append(L)
                t["world"]["W"].append(W)
                t["world"]["corners"].append([list(c) for c in world_corners])

        return {
            "scene_name": scene_name,
            "lidar_slice": lidar_slice,
            "source_field": source_field,
            "frame_indices": [int(f) for f in frame_idxs],
            "lidar_sample_ids": list(sample_ids),
            "m_frame_timestamps": list(m_ts),
            "ego_world": {
                "x": ego_world_x,
                "y": ego_world_y,
                "yaw": ego_world_yaw,
            },
            "instances": sorted(tracks.values(), key=lambda d: d["label"]),
        }


# -----------------------------------------------------------------------------
# 3. build_trajectories — the user-facing operator
# -----------------------------------------------------------------------------

def _record_to_sample(record: TrajectoryRecord, filepath: str) -> fo.Sample:
    sample = fo.Sample(filepath=filepath, tags=[record.kind, record.scene_name])
    for field in SAMPLE_SCHEMA:
        sample[field] = getattr(record, field)
    return sample


def _create_or_overwrite(name: str, overwrite: bool) -> fo.Dataset:
    if name in fo.list_datasets():
        if not overwrite:
            raise RuntimeError(
                f"Target dataset {name!r} already exists; pass overwrite=True"
            )
        fo.delete_dataset(name)
    return fo.Dataset(name=name, persistent=True)


def _build_trajectories(
    *, source: str, target: str, trajectory_root: str, overwrite: bool,
) -> fo.Dataset:
    """Core build logic; the operator wraps this.

    Each trajectory is rendered to a single PNG (BEV plot) via
    matplotlib at build time; ``sample.filepath`` points at the PNG
    and FO's built-in image renderer handles the grid + modal. No
    PyArrow / Parquet / custom JS renderer in the loop.
    """
    src = fo.load_dataset(source)
    tgt = _create_or_overwrite(target, overwrite=overwrite)
    declare_schema(tgt)
    set_sidebar_groups(tgt)

    scenes = sorted(src.distinct("scene_name"))
    ego_size = list(
        (src.info or {}).get("ego_size_lwh_m") or DEFAULT_EGO_SIZE_LWH
    )
    print(f"[build_trajectories] source={source} target={target} scenes={scenes} "
          f"ego_size_lwh_m={ego_size}")

    tgt.info = {
        "source_dataset": source,
        "trajectory_kinds": ["object", "ego"],
        "ego_size_lwh_m": ego_size,
        "coord_frames": ["base", "world", "scene_local", "origin_normalized"],
        "heading_classes": [
            "straight", "slight_left", "slight_right", "left", "right", "u_turn",
        ],
        "quadrants_base": ["front_left", "front_right", "back_left", "back_right"],
        "trajectory_root": trajectory_root,
    }
    tgt.save()

    tmpdir = tempfile.mkdtemp(prefix=f"trajectories-{target}-")
    print(f"[build_trajectories] rendering BEV thumbnails to {tmpdir}")

    samples: list[fo.Sample] = []
    for scene_name in scenes:
        lidar_view = (
            src.select_group_slices("lidar")
               .match(F("scene_name") == scene_name)
               .sort_by("frame_idx")
        )
        ego_record, object_records = build_track_records(
            lidar_view, source_dataset=source, ego_size_lwh=tuple(ego_size),
        )
        records = ([ego_record] if ego_record is not None else []) + list(object_records)
        print(f"[build_trajectories]   {scene_name}: "
              f"ego={'yes' if ego_record else 'no'} "
              f"objects={len(object_records)} (total {len(records)})")
        for record in records:
            local_path = os.path.join(tmpdir, f"{record.output_stem}.png")
            render_trajectory_thumbnail(
                record, local_path, ego_size_lwh=tuple(ego_size),
            )
            samples.append(_record_to_sample(record, local_path))

    print(f"[build_trajectories] adding {len(samples)} samples to {target}")
    tgt.add_samples(samples)
    tgt.compute_metadata()

    remote_dir = f"{trajectory_root.rstrip('/')}/{target}"
    print(f"[build_trajectories] uploading thumbnails to {remote_dir}")
    fos.upload_media(tgt, remote_dir, update_filepaths=True, overwrite=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[build_trajectories] done: {len(tgt)} trajectories in {target!r}")
    return tgt


class BuildTrajectories(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="build_trajectories",
            label="Build trajectories dataset",
            description=(
                "Build a per-trajectory FiftyOne dataset (one sample per "
                "(scene, FO instance)) from a grouped tracking dataset. "
                "Each sample's filepath is a Parquet file with the full "
                "per-frame trajectory data; sample-level fields are "
                "filter-friendly scalars grouped under Identity / Coverage "
                "/ Position / Motion / Shape / QC."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        # Source dataset is the currently-loaded one by default; let the
        # user pick a different one if needed.
        inputs.str(
            "source",
            label="Source tracking dataset",
            description="A grouped dataset produced by one of the loaders "
                        "in fiftyone-tracking-loaders (lidar slice carries "
                        "the cuboid detections).",
            default=ctx.dataset.name if ctx.dataset is not None else None,
            required=True,
        )
        inputs.str(
            "target",
            label="Target trajectories dataset",
            description="Name for the new trajectories dataset.",
            required=True,
        )
        inputs.str(
            "trajectory_root",
            label="Trajectory root (storage URI)",
            description="Where to write per-trajectory Parquet files. "
                        "Local path or any FiftyOne-supported URI (gs://, "
                        "s3://, ...). For shared FOE deployments use a "
                        "remote URI that the App can resolve.",
            default="./trajectories",
        )
        inputs.bool(
            "overwrite",
            label="Overwrite target if it exists",
            default=False,
        )
        return types.Property(
            inputs, view=types.View(label="Build trajectories dataset")
        )

    def execute(self, ctx) -> dict[str, Any]:
        source = ctx.params["source"]
        target = ctx.params["target"]
        trajectory_root = ctx.params.get("trajectory_root") or "./trajectories"
        overwrite = bool(ctx.params.get("overwrite", False))
        ds = _build_trajectories(
            source=source, target=target,
            trajectory_root=trajectory_root, overwrite=overwrite,
        )
        return {
            "target": target,
            "n_trajectories": len(ds),
            "trajectory_root": trajectory_root,
        }


# -----------------------------------------------------------------------------
# 4. get_camera_frame_urls — per-(scene, camera) image URLs by frame_idx
# -----------------------------------------------------------------------------

class GetCameraFrameUrls(foo.Operator):
    """Return ``{frame_idx: media_url}`` for one camera slice of a scene.

    Powers the BEV panel's inline camera-mirror thumbnail. JS fires this
    when the user picks a camera slice; the result is cached client-side
    and indexed by the current scrubber frame_idx to update the inline
    image on every scrub.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_camera_frame_urls",
            label="Get camera frame URLs",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("scene_name", required=True)
        inputs.str("camera_slice", required=True)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        scene_name = ctx.params["scene_name"]
        camera_slice = ctx.params["camera_slice"]
        slices = list(ctx.dataset.group_slices or [])
        if camera_slice not in slices:
            return {
                "scene_name": scene_name, "camera_slice": camera_slice,
                "frame_urls": {},
                "error": (
                    f"slice {camera_slice!r} not in group_slices "
                    f"{slices!r}"
                ),
            }
        view = ctx.dataset.select_group_slices(camera_slice).match(
            F("scene_name") == scene_name
        ).sort_by("frame_idx")
        frame_idxs, filepaths = view.values(["frame_idx", "filepath"])
        # Resolve each filepath into a browser-loadable URL — fos.get_url
        # returns a signed URL for gs:// / s3:// paths, or the raw path
        # for already-HTTP / local FS paths. Cached client-side per
        # (scene, slice); the JS layer doesn't re-fire unless the user
        # changes either.
        frame_urls = {}
        for fi, fp in zip(frame_idxs, filepaths):
            if fi is None or not fp:
                continue
            try:
                url = fos.get_url(fp)
            except Exception:
                url = fp  # fallback; browser may not be able to load it
            frame_urls[str(int(fi))] = url
        return {
            "scene_name": scene_name,
            "camera_slice": camera_slice,
            "frame_urls": frame_urls,
        }


# -----------------------------------------------------------------------------
# 5. view_track_patches — apply a patches view filtered to one track
# -----------------------------------------------------------------------------

class ViewTrackPatches(foo.Operator):
    """Switch the App view to one-patch-per-detection for one or more tracks.

    Flattens the dataset across one or more camera slices, filters to
    a single scene + a set of FO ``Instance.id`` values (the FO-side
    cross-frame identifier shared by every Detection of the same
    track), sorts the resulting patches by a configurable order
    field, and applies the view via ``ctx.ops.set_view``.

    Why ``instance._id`` instead of the source-side ``tracking_id``?
    The source-side field name varies across loaders + external
    datasets and its storage type isn't uniform (str vs int). The FO
    ``Instance`` id is dataset-agnostic and the same for all of a
    track's Detections regardless of loader. The BEV panel passes
    the selected instance hexes directly.

    The scene-match field and the post-flatten order field are both
    configurable so this operator works against datasets that use a
    different schema (``sequence``/``run`` instead of ``scene_name``;
    ``frame_number``/``timestamp_s`` instead of ``frame_idx``).

    Inputs:
      scene_name: str — required value to filter the scene field on.
      scene_field: str (default ``"scene_name"``) — the sample-level
        string field that holds the scene/run/sequence identifier.
      instance_ids: list[str] — one FO ``Instance`` hex per track to
        include. ``instance_id`` (singular str) is also accepted as a
        single-track shortcut for backwards-compat.
      camera_slices: list[str] (or comma-separated string) —
        optional. When omitted/empty, the operator auto-picks every
        non-lidar group slice (i.e. all cameras). ``camera_slice``
        (singular str) is also accepted.
      order_field: str (default ``"frame_idx"``) — sample-level field
        the patches view is sorted by. Preserved on the patches via
        ``to_patches(other_fields=[order_field])``. Set to empty
        string to skip sorting.
      match_field, match_value: optional advanced override. When set,
        replaces the ``detections.instance._id`` match with
        ``F(match_field) == match_value`` (also matching the int form
        if it parses). Useful for filtering by
        ``detections.tracking_id`` or a custom dynamic field.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_track_patches",
            label="View track patches",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("scene_name", required=True)
        inputs.str("scene_field", required=False)
        inputs.list("instance_ids", types.String(), required=False)
        inputs.str("instance_id", required=False)
        inputs.list("camera_slices", types.String(), required=False)
        inputs.str("camera_slice", required=False)
        inputs.str("order_field", required=False)
        inputs.str("match_field", required=False)
        inputs.str("match_value", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        from bson import ObjectId

        scene_name = ctx.params["scene_name"]
        scene_field = ctx.params.get("scene_field") or "scene_name"
        order_field = ctx.params.get("order_field")
        if order_field is None:
            order_field = "frame_idx"
        all_slices = list(ctx.dataset.group_slices or [])

        # ----- Resolve which slices to flatten -----
        slices_param = ctx.params.get("camera_slices")
        legacy_single = ctx.params.get("camera_slice")
        if slices_param:
            if isinstance(slices_param, str):
                slices = [s.strip() for s in slices_param.split(",") if s.strip()]
            else:
                slices = list(slices_param)
        elif legacy_single:
            slices = [legacy_single]
        else:
            slices = [s for s in all_slices if not s.startswith("lidar")]

        missing = [s for s in slices if s not in all_slices]
        if missing:
            return {
                "error": (
                    f"slices {missing!r} not in dataset.group_slices "
                    f"{all_slices!r}"
                )
            }
        if not slices:
            return {"error": "no camera slices to view"}

        # ----- Resolve the per-patch identity filter -----
        match_field = ctx.params.get("match_field")
        match_value = ctx.params.get("match_value")
        ids_param = ctx.params.get("instance_ids")
        legacy_single_id = ctx.params.get("instance_id")

        if match_field:
            if match_value is None:
                return {"error": "match_field set but match_value missing"}
            try:
                tid_int = int(match_value)
                identity_filter = (
                    (F(match_field) == match_value)
                    | (F(match_field) == tid_int)
                )
            except (ValueError, TypeError):
                identity_filter = F(match_field) == match_value
            identity_desc = f"{match_field}={match_value!r}"
        else:
            # Resolve the FO Instance hex list.
            if ids_param:
                hexes = (
                    [h.strip() for h in ids_param.split(",") if h.strip()]
                    if isinstance(ids_param, str)
                    else [str(h) for h in ids_param]
                )
            elif legacy_single_id:
                hexes = [str(legacy_single_id)]
            else:
                return {
                    "error": "one of instance_ids / instance_id / match_field is required"
                }
            try:
                oids = [ObjectId(h) for h in hexes]
            except Exception as e:
                return {"error": f"invalid instance hex in {hexes!r}: {e!r}"}
            if len(oids) == 1:
                identity_filter = F("detections.instance._id") == oids[0]
            else:
                identity_filter = F("detections.instance._id").is_in(oids)
            identity_desc = (
                f"instance._id ∈ {{{', '.join(hexes)}}}"
                if len(hexes) > 1
                else f"instance._id={hexes[0]}"
            )

        # to_patches preserves selected sample-level fields on the
        # patches via other_fields. We add the order_field so the
        # final sort_by can find it (default frame_idx; user override
        # like "timestamp_s"/"frame_number" works too). Empty
        # order_field skips both preservation + sort.
        other_fields = []
        if order_field:
            other_fields.append(order_field)

        view = (
            ctx.dataset.select_group_slices(slices)
                .match(F(scene_field) == scene_name)
                .to_patches("detections", other_fields=other_fields)
                .match(identity_filter)
        )
        if order_field:
            view = view.sort_by(order_field)

        n = view.count()
        if n == 0:
            return {
                "error": (
                    f"no patches for {identity_desc} on slices "
                    f"{slices!r} where {scene_field}={scene_name!r}"
                )
            }

        ctx.ops.set_view(view=view)
        return {
            "scene_field": scene_field,
            "scene_name": scene_name,
            "identity": identity_desc,
            "camera_slices": slices,
            "order_field": order_field or None,
            "n_patches": int(n),
        }


# -----------------------------------------------------------------------------
# Track correction: split / join helpers
# -----------------------------------------------------------------------------

def _resolve_track_edit_targets(ctx, target_field):
    """Pull ``(instance_hexes, scene_name)`` from whichever invocation
    surface fired the operator.

    Priority:
      1. ``ctx.params["instance_ids"]`` provided directly (BEV panel).
      2. ``ctx.selected_labels`` (embeddings panel lasso).
      3. ``ctx.selected`` (grid sample selection).

    Scene name comes from ``ctx.params["scene_name"]`` if present, else
    inferred from the selection. Returns ``"__MULTI__"`` for the scene
    name when the selection spans multiple scenes; the caller should
    surface that as an error.
    """
    instance_hexes = list(ctx.params.get("instance_ids") or [])
    if ctx.params.get("instance_id"):  # legacy single
        instance_hexes = instance_hexes + [ctx.params["instance_id"]]
    scenes: set[str] = set()

    if not instance_hexes:
        # PatchesView first: ctx.selected holds patch ids that don't
        # resolve via dataset.select(...).
        patches_hexes, _pmin, patches_scenes, _pfield = (
            _instances_from_patches_view(ctx)
        )
        if patches_hexes:
            instance_hexes = patches_hexes
            scenes = patches_scenes or set()
        elif ctx.selected_labels:
            instance_hexes, _frame, scenes = (
                _instances_from_selected_labels(
                    ctx.dataset, ctx.selected_labels, target_field
                )
            )
        elif ctx.selected:
            instance_hexes, _frame, scenes = (
                _instances_from_selected_samples(
                    ctx.dataset, ctx.selected, target_field
                )
            )

    scene_name = ctx.params.get("scene_name")
    if scene_name is None:
        if len(scenes) == 1:
            scene_name = next(iter(scenes))
        elif len(scenes) > 1:
            scene_name = "__MULTI__"
    return instance_hexes, scene_name


def _resolve_form_defaults(ctx):
    """Inspect ``ctx`` once and return form-prefill hints for the
    Split/Merge operator inputs.

    Returns a dict with keys:
      - ``detection_fields``: sorted list of Detections-typed field names
        on the lidar slice. Empty if discovery fails.
      - ``inferred_scene``: str or None (when all selected samples/labels
        share a single scene_name).
      - ``inferred_instance_ids``: list[str] hexes derived from the
        selection (via the field the selection came from).
      - ``selection_summary``: short string describing what's selected
        (used in field descriptions to surface context to the user).
      - ``selection_frame_range``: ``(min, max)`` tuple of ``frame_idx``
        across selected samples, or None.
      - ``selection_field``: the Detections field the selection came
        from (used as the source_field default).
    """
    # Detections fields on the lidar slice
    detection_fields: list[str] = []
    try:
        lidar_slice = _resolve_lidar_slice(ctx.dataset)
        if lidar_slice:
            schema = ctx.dataset.select_group_slices(
                lidar_slice
            ).get_field_schema(
                ftype=fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Detections,
            )
            detection_fields = sorted(schema.keys())
    except Exception:
        detection_fields = []

    inferred_scene = None
    inferred_instance_ids: list[str] = []
    selection_summary = ""
    selection_frame_range: tuple | None = None
    selection_field = None

    selected_labels = list(ctx.selected_labels or [])
    selected_samples = list(ctx.selected or [])

    # Patches view takes priority — selection IDs there are patch ids,
    # not dataset sample ids, so they must be looked up via the view.
    patches_hexes, patches_min, patches_scenes, patches_field = (
        _instances_from_patches_view(ctx)
    )

    if patches_hexes is not None:
        selection_field = patches_field
        inferred_instance_ids = patches_hexes
        if patches_scenes and len(patches_scenes) == 1:
            inferred_scene = next(iter(patches_scenes))
        selection_summary = (
            f"{len(selected_samples)} patch(es) selected "
            f"(from PatchesView of field {patches_field!r})"
        )
        if patches_min is not None:
            try:
                view = ctx.view
                hi = view.select(selected_samples).bounds("frame_idx")[1]
                selection_frame_range = (
                    patches_min, int(hi) if hi is not None else patches_min
                )
            except Exception:
                selection_frame_range = (patches_min, patches_min)
    elif selected_labels:
        selection_field = selected_labels[0].get("field") or DEFAULT_SOURCE_FIELD
        instances, _frame, scenes = _instances_from_selected_labels(
            ctx.dataset, selected_labels, selection_field
        )
        inferred_instance_ids = instances
        if len(scenes) == 1:
            inferred_scene = next(iter(scenes))
        selection_summary = (
            f"{len(selected_labels)} label(s) selected "
            f"(from field {selection_field!r})"
        )
        sample_ids = sorted({sl["sample_id"] for sl in selected_labels})
        try:
            lo, hi = ctx.dataset.select(sample_ids).bounds("frame_idx")
            if lo is not None:
                selection_frame_range = (int(lo), int(hi))
        except Exception:
            pass
    elif selected_samples:
        selection_field = DEFAULT_SOURCE_FIELD
        instances, _frame, scenes = _instances_from_selected_samples(
            ctx.dataset, selected_samples, selection_field
        )
        inferred_instance_ids = instances
        if len(scenes) == 1:
            inferred_scene = next(iter(scenes))
        selection_summary = f"{len(selected_samples)} sample(s) selected"
        try:
            lo, hi = ctx.dataset.select(selected_samples).bounds("frame_idx")
            if lo is not None:
                selection_frame_range = (int(lo), int(hi))
        except Exception:
            pass

    return {
        "detection_fields": detection_fields,
        "inferred_scene": inferred_scene,
        "inferred_instance_ids": inferred_instance_ids,
        "selection_summary": selection_summary,
        "selection_frame_range": selection_frame_range,
        "selection_field": selection_field,
    }


class SplitTrack(foo.Operator):
    """Split one track at a frame.

    Frames with ``frame_idx < split_frame`` keep the original
    ``instance._id``; everything at or after ``split_frame`` is moved
    to a freshly minted ``fo.Instance()``. Mutations apply across all
    group slices (lidar + cameras) so cross-slice ids stay consistent.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="split_track",
            label="Split track at frame",
        )

    def resolve_input(self, ctx):
        d = _resolve_form_defaults(ctx)
        sel = d["selection_summary"] or "(nothing selected)"
        frame_range = d["selection_frame_range"]
        frame_hint = (
            f" Selection spans frame_idx {frame_range[0]}..{frame_range[1]}."
            if frame_range else ""
        )

        inputs = types.Object()

        inputs.str(
            "scene_name",
            label="Scene name",
            description=(
                "Scene to apply the split to. Auto-derived from the "
                f"selection when possible. Current: {sel}."
            ),
            default=d["inferred_scene"],
            required=False,
        )

        src_default = (
            d["selection_field"]
            or (DEFAULT_SOURCE_FIELD
                if DEFAULT_SOURCE_FIELD in d["detection_fields"]
                else (d["detection_fields"][0] if d["detection_fields"]
                      else DEFAULT_SOURCE_FIELD))
        )
        src_desc = (
            "Detections field containing the track to split. The "
            "instance hex you pick below must exist in this field."
        )
        if d["detection_fields"]:
            src_desc += " Detections fields on this dataset: " + ", ".join(
                d["detection_fields"]
            ) + "."
        inputs.str(
            "source_field",
            label="Source field",
            description=src_desc,
            default=src_default,
            required=True,
        )

        inputs.str(
            "target_field",
            label="Target field",
            description=(
                "Field to write the split into. Cloned from the source "
                f"field on first edit if it doesn't exist. Default "
                f"{DEFAULT_TARGET_FIELD!r} keeps the source intact."
            ),
            default=DEFAULT_TARGET_FIELD,
            required=True,
        )

        inputs.list(
            "instance_ids",
            types.String(),
            label="Instance IDs",
            description=(
                "FO ``instance._id`` hex(es) to split. Auto-derived from "
                "the selection; split_track expects exactly one. Hexes "
                f"derived from current selection: {d['inferred_instance_ids']!r}."
            ),
            default=d["inferred_instance_ids"] or None,
            required=False,
        )

        inputs.int(
            "split_frame",
            label="Split frame",
            description=(
                "Frame index to split at. Detections with frame_idx < "
                "this keep the original instance; detections at or after "
                "get a freshly minted instance." + frame_hint
            ),
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        source_field = ctx.params.get("source_field") or DEFAULT_SOURCE_FIELD
        target_field = ctx.params.get("target_field") or DEFAULT_TARGET_FIELD
        _ensure_target_field(ctx.dataset, source_field, target_field)

        instance_hexes, scene_name = _resolve_track_edit_targets(
            ctx, target_field
        )
        split_frame = ctx.params.get("split_frame")

        if scene_name == "__MULTI__":
            return {"error": "split_track: selection spans multiple scenes"}
        if not instance_hexes:
            return {"error": "split_track: no instance to split"}
        if len(instance_hexes) != 1:
            return {
                "error": (
                    f"split_track expects exactly 1 instance; got "
                    f"{len(instance_hexes)}"
                )
            }
        if split_frame is None:
            return {"error": "split_track: split_frame is required"}

        instance_hex = instance_hexes[0]
        oid = ObjectId(instance_hex)
        split_frame = int(split_frame)

        # Pre-flight: what frames does this instance actually occupy in
        # target_field on the lidar slice? Surface that range so the user
        # can spot "I picked a fragment / wrong instance" misclicks.
        lidar_slice = _resolve_lidar_slice(ctx.dataset)
        instance_frame_range = None
        if lidar_slice is not None:
            lidar_for_inst = ctx.dataset.select_group_slices(lidar_slice)
            if scene_name:
                lidar_for_inst = lidar_for_inst.match(
                    F("scene_name") == scene_name
                )
            lidar_for_inst = lidar_for_inst.match({
                f"{target_field}.detections.instance._id": oid,
            })
            lo, hi = lidar_for_inst.bounds("frame_idx")
            n_lidar = lidar_for_inst.count()
            instance_frame_range = {
                "min": None if lo is None else int(lo),
                "max": None if hi is None else int(hi),
                "n_lidar_samples": int(n_lidar),
            }

        new_instance = fo.Instance()
        n_det, n_samp, per_slice = _reassign_instances_across_slices(
            ctx.dataset,
            target_field=target_field,
            scene_name=scene_name,
            match_instance_oids=[oid],
            replace_when_hex_in={instance_hex},
            new_instance=new_instance,
            extra_match=(F("frame_idx") >= split_frame),
            log_prefix=f"split_track frame>={split_frame}",
        )

        # No-op surface: split_frame lay outside the instance's frame
        # range, so the new instance has no detections. Return that
        # explicitly so the panel doesn't move selection to a phantom.
        if n_det == 0:
            return {
                "noop": True,
                "reason": (
                    f"instance {instance_hex} has no detections at "
                    f"frame_idx >= {split_frame}"
                ),
                "scene_name": scene_name,
                "source_field": source_field,
                "target_field": target_field,
                "split_frame": split_frame,
                "old_instance_id": instance_hex,
                "instance_frame_range": instance_frame_range,
                "per_slice": per_slice,
            }

        return {
            "scene_name": scene_name,
            "source_field": source_field,
            "target_field": target_field,
            "split_frame": split_frame,
            "old_instance_id": instance_hex,
            "new_instance_id": new_instance.id,
            "n_detections_reassigned": n_det,
            "n_samples_touched": n_samp,
            "instance_frame_range": instance_frame_range,
            "per_slice": per_slice,
        }


class JoinTracks(foo.Operator):
    """Merge N tracks onto a single ``fo.Instance``.

    The winning instance is the one whose earliest detection has the
    lowest ``frame_idx`` on the lidar slice. All other selected
    instances are rewritten to the winner across every group slice.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="join_tracks",
            label="Join tracks",
        )

    def resolve_input(self, ctx):
        d = _resolve_form_defaults(ctx)
        sel = d["selection_summary"] or "(nothing selected)"

        inputs = types.Object()

        inputs.str(
            "scene_name",
            label="Scene name",
            description=(
                "Scene to apply the merge to. Auto-derived from the "
                f"selection when possible. Current: {sel}."
            ),
            default=d["inferred_scene"],
            required=False,
        )

        src_default = (
            d["selection_field"]
            or (DEFAULT_SOURCE_FIELD
                if DEFAULT_SOURCE_FIELD in d["detection_fields"]
                else (d["detection_fields"][0] if d["detection_fields"]
                      else DEFAULT_SOURCE_FIELD))
        )
        src_desc = (
            "Detections field containing the tracks to merge. The "
            "instance hexes you pick below must exist in this field."
        )
        if d["detection_fields"]:
            src_desc += " Detections fields on this dataset: " + ", ".join(
                d["detection_fields"]
            ) + "."
        inputs.str(
            "source_field",
            label="Source field",
            description=src_desc,
            default=src_default,
            required=True,
        )

        inputs.str(
            "target_field",
            label="Target field",
            description=(
                "Field to write the merge into. Cloned from the source "
                f"field on first edit if it doesn't exist. Default "
                f"{DEFAULT_TARGET_FIELD!r} keeps the source intact."
            ),
            default=DEFAULT_TARGET_FIELD,
            required=True,
        )

        inputs.list(
            "instance_ids",
            types.String(),
            label="Instance IDs",
            description=(
                "FO ``instance._id`` hexes to merge. Auto-derived from "
                "the selection; merge expects 2 or more. The instance "
                "with the earliest frame_idx wins; all others are "
                "rewritten onto it. Hexes derived from current "
                f"selection: {d['inferred_instance_ids']!r}."
            ),
            default=d["inferred_instance_ids"] or None,
            required=False,
        )

        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        source_field = ctx.params.get("source_field") or DEFAULT_SOURCE_FIELD
        target_field = ctx.params.get("target_field") or DEFAULT_TARGET_FIELD
        _ensure_target_field(ctx.dataset, source_field, target_field)

        instance_hexes, scene_name = _resolve_track_edit_targets(
            ctx, target_field
        )
        if scene_name == "__MULTI__":
            return {"error": "join_tracks: selection spans multiple scenes"}
        if len(instance_hexes) < 2:
            return {
                "noop": True,
                "reason": "fewer than 2 instances to join",
                "n_instances": len(instance_hexes),
            }

        # --- Choose the winner: earliest lidar appearance
        # Same elemMatch caveat as split: F(nested.array.field) == oid
        # does not reliably hit array elements; use the Mongo dict form.
        lidar_slice = _resolve_lidar_slice(ctx.dataset)
        if lidar_slice is None:
            return {"error": "no lidar group slice on this dataset"}
        lidar = ctx.dataset.select_group_slices(lidar_slice)
        if scene_name:
            lidar = lidar.match(F("scene_name") == scene_name)

        INF = float("inf")
        per_inst_min: dict[str, float] = {}
        for hex_ in instance_hexes:
            sub = lidar.match({
                f"{target_field}.detections.instance._id": ObjectId(hex_),
            })
            lo, _hi = sub.bounds("frame_idx")
            per_inst_min[hex_] = INF if lo is None else float(lo)
        if all(v == INF for v in per_inst_min.values()):
            return {
                "error": "no detections found for any selected instance",
                "per_inst_min": {k: (None if v == INF else v)
                                 for k, v in per_inst_min.items()},
                "target_field": target_field,
                "scene_name": scene_name,
            }
        winner_hex = min(per_inst_min, key=per_inst_min.get)
        loser_hexes = [h for h in instance_hexes if h != winner_hex]

        # --- Find the existing fo.Instance object for the winner
        # (must reuse the same object across all loser detections so they
        # share an id; minting fresh fo.Instance()s would mint fresh ids).
        # Look up on the lidar slice via dict-form match for reliable
        # elemMatch on the nested array path.
        winner_oid = ObjectId(winner_hex)
        winner_instance = None
        for sample in (
            lidar.match({
                f"{target_field}.detections.instance._id": winner_oid,
            }).limit(1)
        ):
            dets = sample[target_field]
            if dets is None:
                continue
            for det in dets.detections:
                if det.instance is not None and det.instance.id == winner_hex:
                    winner_instance = det.instance
                    break
            if winner_instance is not None:
                break
        if winner_instance is None:
            return {"error": f"could not locate winner instance {winner_hex}"}

        loser_oids = [ObjectId(h) for h in loser_hexes]
        loser_hex_set = set(loser_hexes)
        n_det, n_samp, per_slice = _reassign_instances_across_slices(
            ctx.dataset,
            target_field=target_field,
            scene_name=scene_name,
            match_instance_oids=loser_oids,
            replace_when_hex_in=loser_hex_set,
            new_instance=winner_instance,
            log_prefix=f"join_tracks winner={winner_hex}",
        )

        return {
            "scene_name": scene_name,
            "source_field": source_field,
            "target_field": target_field,
            "winner_instance_id": winner_hex,
            "merged": loser_hexes,
            "n_detections_reassigned": n_det,
            "n_samples_touched": n_samp,
            "per_slice": per_slice,
        }


# -----------------------------------------------------------------------------
# Plugin registration
# -----------------------------------------------------------------------------

def register(p):
    p.register(ListTrackingScenes)
    p.register(GetSceneTrackPayload)
    p.register(GetCameraFrameUrls)
    p.register(ViewTrackPatches)
    p.register(BuildTrajectories)
    p.register(SplitTrack)
    p.register(JoinTracks)
