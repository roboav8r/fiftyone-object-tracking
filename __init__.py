"""Operators for the FiftyOne object-tracking toolkit plugin.

Powers the ``ObjectTracking`` panel. Operators exposed through
``register()``:

  - ``list_tracking_scenes``    (unlisted) — enumerate scenes +
                                group-slice names for the Scene tab
  - ``get_scene_track_payload`` (unlisted) — one-shot per-scene
                                trajectory bundle for the Scene tab
  - ``get_camera_frame_urls``   (unlisted) — per-(scene, camera)
                                frame URLs for the camera mirror
  - ``view_track_patches``      (unlisted) — flatten the App view to
                                one-patch-per-detection for a track
  - ``build_trajectories``      (listed)   — extract ephemeral
                                tracklets from the tracking dataset
                                into an ExecutionStore (no dataset is
                                created); the Trajectories tab reads
                                them back via ``get_trajectories``
  - ``get_trajectories``        (unlisted) — read built tracklets (+
                                the current filter selection) for the
                                Trajectories tab grid
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F

from ._clustering import cluster_from_matrix, tracks_to_arrays
from ._dtw import pairwise_dtw_matrix
from ._records import (
    DEFAULT_EGO_SIZE_LWH,
    TrajectoryRecord,
    build_track_records,
)


DEFAULT_LIDAR_SLICE = "lidar"


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


def _resolve_lidar_slice(group_slices) -> str | None:
    """Pick the lidar group slice: exact ``lidar`` else first ``*lidar*``."""
    slices = list(group_slices or [])
    if DEFAULT_LIDAR_SLICE in slices:
        return DEFAULT_LIDAR_SLICE
    return next((s for s in slices if "lidar" in s.lower()), None)


def _detection_slices(ds) -> list[str]:
    """Group slices whose flattened schema has a ``detections`` field.

    Lidar cuboids + camera 2D boxes both carry the same FO ``instance``,
    so tag write-through hits the track wherever it's viewed.
    """
    out: list[str] = []
    for sl in list(ds.group_slices or []):
        try:
            if "detections" in ds.select_group_slices(sl).get_field_schema():
                out.append(sl)
        except Exception:
            continue
    return out


def _instance_label_tags(lidar_view) -> dict[str, set]:
    """Map ``instance._id`` hex -> union of its detection label tags.

    Read in one aggregation over the scene's lidar detections so
    build_trajectories can re-hydrate the tags applied by
    tag_trajectories (the durable source of truth is the label tag).

    Values are read *without* ``unwind`` so each sample yields a list
    over its detections: ``iids_per_sample[i][j]`` is detection j's
    ``instance._id`` and ``tags_per_sample[i][j]`` is its list of tags.
    (Unwinding would flatten the per-detection ``tags`` list an extra
    level and misalign it against the instance ids.)
    """
    iids_per_sample = lidar_view.values("detections.detections.instance._id")
    tags_per_sample = lidar_view.values("detections.detections.tags")
    out: dict[str, set] = {}
    for sample_iids, sample_tags in zip(iids_per_sample, tags_per_sample):
        if not sample_iids:
            continue
        for iid, tags in zip(sample_iids, sample_tags or []):
            if iid is None:
                continue
            bucket = out.setdefault(str(iid), set())
            if tags:
                bucket.update(tags)
    return out


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

        view = ctx.view.select_group_slices(lidar_slice) \
            if hasattr(ctx, "view") and ctx.view is not None \
            else ctx.dataset.select_group_slices(lidar_slice)

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

        return {
            "scenes": sorted(scenes.values(), key=lambda s: s["scene_name"]),
            "group_slices": {"all": group_slices, "lidar": lidar_slice},
        }


class ResolveSceneForSample(foo.Operator):
    """Return the ``scene_name`` for a given id, so the modal panel can infer
    its scene from the open sample. The modal's active id may be any slice's
    sample (e.g. a camera, not lidar) or the group id, so match on either
    ``_id`` or ``group._id`` across every group slice (a single indexed lookup
    per slice, with early exit)."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="resolve_scene_for_sample",
            label="Resolve scene for sample",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sample_id", required=True)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        from bson import ObjectId

        raw = ctx.params.get("sample_id")
        try:
            oid = ObjectId(raw)
        except Exception:
            return {"scene_name": None}

        ds = ctx.dataset
        match = (F("_id") == oid) | (F("group._id") == oid)
        for sl in list(ds.group_slices or []):
            try:
                vals = (
                    ds.select_group_slices(sl).match(match).limit(1)
                    .values("scene_name")
                )
            except Exception:
                continue
            if vals and vals[0]:
                return {"scene_name": vals[0], "sample_id": str(raw)}
        return {"scene_name": None, "sample_id": str(raw)}


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
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        scene_name = ctx.params["scene_name"]
        lidar_slice = ctx.params.get("lidar_slice") or DEFAULT_LIDAR_SLICE

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
            "detections.detections.instance._id",
            "detections.detections.label",
            "detections.detections.tracking_id",
            "detections.detections.location",
            "detections.detections.rotation",
            "detections.detections.dimensions",
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

# Name of the dataset-scoped ExecutionStore that holds built tracklets +
# the current filter selection. The Trajectories tab reads it back via
# get_trajectories; nothing is persisted as FiftyOne samples.
STORE_NAME = "object_tracking"

# Scalar fields copied verbatim from a TrajectoryRecord into a tracklet
# dict. Per-frame arrays are handled separately (downsampled to XY).
_TRACKLET_SCALARS = (
    "kind", "instance_id", "tracking_id", "tracking_name", "scene_name",
    "n_frames", "duration_s", "frame_idx_first", "frame_idx_last",
    "is_fragmented", "n_fragments", "is_stationary",
    "start_quadrant_base", "end_quadrant_base",
    "start_distance_m_base", "end_distance_m_base", "closest_approach_m_base",
    "displacement_m", "path_length_m", "straightness",
    "heading_change_deg", "heading_class", "side_pass", "crosses_ego_path",
    "mean_speed_m_s", "max_speed_m_s",
    # QC
    "n_distinct_classes", "tracking_names_distinct",
    "max_step_jump_m", "max_gap_s",
)


def _json_scalar(v):
    """Coerce numpy/str/bool scalars to JSON-able Python values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_json_scalar(x) for x in v]
    return v


def _xy_list(arr) -> list[list[float]]:
    """First two columns of an Nx2/Nx3 array as a JSON-able list of [x, y]."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return []
    return [[float(row[0]), float(row[1])] for row in a]


def _record_to_tracklet(record: TrajectoryRecord, track_idx: int) -> dict:
    """Serialize one TrajectoryRecord to a JSON-able tracklet dict.

    ``track_idx`` is the trajectory's index within its scene's record
    list (ego first when present) — the identifier the filter operator
    returns as ``{scene_name: [track_idx, ...]}``.
    """
    out = {"track_idx": int(track_idx)}
    for name in _TRACKLET_SCALARS:
        out[name] = _json_scalar(getattr(record, name))
    out["tracking_id"] = str(out["tracking_id"])
    # User-applied trajectory tags; re-hydrated from the underlying
    # detection label tags by build_trajectories (see _instance_label_tags).
    out["tags"] = []
    # Per-frame XY for the in-panel mini-BEV thumbnail (base + world; ego
    # also carries scene-local so its path isn't a degenerate dot).
    out["frame_indices"] = [int(x) for x in np.asarray(record.frame_indices).tolist()]
    out["fragment_ids"] = [int(x) for x in np.asarray(record.fragment_ids).tolist()]
    out["xy_base"] = _xy_list(record.translations_base)
    out["xy_world"] = _xy_list(record.translations_world)
    out["xy_scene_local"] = _xy_list(record.translations_scene_local)
    return out


def _scene_choices(ctx):
    """Choices for the scene picker: every scene + an 'All scenes' option."""
    scenes = sorted(ctx.dataset.distinct("scene_name")) if ctx.dataset else []
    choices = types.Choices()
    choices.add_choice("__all__", label="All scenes")
    for s in scenes:
        choices.add_choice(s, label=s)
    return scenes, choices


class BuildTrajectories(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="build_trajectories",
            label="Build trajectories",
            description=(
                "Extract ephemeral per-(scene, instance) tracklets from the "
                "current tracking dataset into an in-memory store. Nothing "
                "is persisted as samples; the Trajectories tab renders the "
                "result as an in-panel grid and the filter operator selects "
                "across it."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        scenes, choices = _scene_choices(ctx)
        default_scene = ctx.params.get("scene") or (
            scenes[0] if scenes else "__all__"
        )
        inputs.enum(
            "scene", choices.values(), default=default_scene, required=True,
            view=choices, label="Scene",
            description="Build trajectories for one scene or all scenes.",
        )
        return types.Property(inputs, view=types.View(label="Build trajectories"))

    def execute(self, ctx) -> dict[str, Any]:
        ds = ctx.dataset
        if ds is None:
            return {"error": "No dataset loaded."}
        lidar_slice = _resolve_lidar_slice(ds.group_slices)
        if lidar_slice is None:
            return {"error": "No lidar group slice found on this dataset."}

        scene_param = ctx.params.get("scene") or "__all__"
        all_scenes = sorted(ds.distinct("scene_name"))
        targets = all_scenes if scene_param == "__all__" else [scene_param]
        ego_size = tuple(
            (ds.info or {}).get("ego_size_lwh_m") or DEFAULT_EGO_SIZE_LWH
        )

        store = ctx.store(STORE_NAME)
        built: list[str] = []
        total = 0
        classes_seen: set[str] = set()
        for scene_name in targets:
            lidar_view = (
                ds.select_group_slices(lidar_slice)
                  .match(F("scene_name") == scene_name)
                  .sort_by("frame_idx")
            )
            ego_record, object_records = build_track_records(
                lidar_view, source_dataset=ds.name, ego_size_lwh=ego_size,
            )
            records = (
                ([ego_record] if ego_record is not None else [])
                + list(object_records)
            )
            tracklets = [
                _record_to_tracklet(r, i) for i, r in enumerate(records)
            ]
            # Re-hydrate trajectory tags from the underlying detection
            # label tags (durable across rebuilds; written by tag_trajectories).
            tag_map = _instance_label_tags(lidar_view)
            for t in tracklets:
                iid = t.get("instance_id")
                if iid and str(iid) in tag_map:
                    t["tags"] = sorted(tag_map[str(iid)])
            for t in tracklets:
                if t.get("kind") == "ego":
                    classes_seen.add("ego")
                elif t.get("tracking_name"):
                    classes_seen.add(str(t["tracking_name"]))
            store.set(f"tracklets:{scene_name}", tracklets)
            built.append(scene_name)
            total += len(tracklets)

        # A new build reassigns track_idx, so it invalidates the prior
        # filter selection AND any cached clusters (their row→track_idx
        # maps no longer line up).
        store.delete("filter_selection")
        for key in list(store.list_keys()):
            if key.startswith("clusters:"):
                store.delete(key)
        store.delete("clusters_meta")
        store.set("meta", {
            "updated": time.time(),
            "source": ds.name,
            "scenes": built,
            "ego_size_lwh_m": list(ego_size),
            "n_trajectories": total,
            # Distinct object classes, so cluster_trajectories' class picker
            # reads them cheaply instead of loading every tracklet.
            "classes": sorted(classes_seen),
        })
        return {"scenes": built, "n_trajectories": total}


# -----------------------------------------------------------------------------
# get_trajectories — read built tracklets (+ filter selection) for the grid
# -----------------------------------------------------------------------------

class GetTrajectories(foo.Operator):
    """Return built tracklets for the Trajectories-tab grid.

    Reads the dataset-scoped store populated by ``build_trajectories``.
    If a ``scene_name`` is given, returns that scene's tracklets; else
    returns tracklets for every built scene. Each tracklet is tagged
    ``_matched`` per the current filter selection (if any).
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_trajectories",
            label="Get trajectories",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("scene_name", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        store = ctx.store(STORE_NAME)
        meta = store.get("meta") or {}
        scene = ctx.params.get("scene_name")
        sel = store.get("filter_selection") or {}
        selection = sel.get("selection") or {}

        if scene:
            scenes = [scene]
        else:
            scenes = list(meta.get("scenes", []))

        tracklets: list[dict] = []
        for s in scenes:
            rows = store.get(f"tracklets:{s}") or []
            matched_idxs = set(selection.get(s) or [])
            for t in rows:
                t["_matched"] = t["track_idx"] in matched_idxs
            tracklets.extend(rows)

        return {
            "scene_name": scene,
            "tracklets": tracklets,
            "meta": meta,
            "updated": meta.get("updated"),
            "filter": {
                "active": bool(sel),
                "summary": sel.get("summary"),
                "spec": sel.get("spec"),
            },
        }


# -----------------------------------------------------------------------------
# filter_trajectories — select tracklets matching one or more conditions
# -----------------------------------------------------------------------------

# Tracklet fields exposed to the condition builder, grouped by value kind so
# the UI / evaluator can pick sensible operators.
_FILTER_NUM_FIELDS = (
    "n_frames", "duration_s", "max_gap_s", "max_step_jump_m",
    "displacement_m", "path_length_m", "straightness",
    "mean_speed_m_s", "max_speed_m_s", "n_distinct_classes", "n_fragments",
    "start_distance_m_base", "end_distance_m_base", "closest_approach_m_base",
)
_FILTER_CAT_FIELDS = (
    "tracking_name", "kind", "heading_class",
    "start_quadrant_base", "end_quadrant_base",
)
_FILTER_BOOL_FIELDS = (
    "is_fragmented", "is_stationary", "side_pass", "crosses_ego_path",
)
_FILTER_FIELDS = _FILTER_NUM_FIELDS + _FILTER_CAT_FIELDS + _FILTER_BOOL_FIELDS
_FILTER_OPS = ("<", "<=", ">", ">=", "==", "!=", "in", "not in")

# Human-readable labels for the field dropdown so users aren't confused by the
# raw field names (notably: `kind` is object/ego, the CLASS lives in
# `tracking_name`).
_FIELD_LABELS = {
    "tracking_name": "Class", "kind": "Kind (object/ego)",
    "heading_class": "Heading", "start_quadrant_base": "Start quadrant",
    "end_quadrant_base": "End quadrant",
    "n_frames": "# frames", "duration_s": "Duration (s)",
    "max_gap_s": "Max gap (s)", "max_step_jump_m": "Max step jump (m)",
    "displacement_m": "Displacement (m)", "path_length_m": "Path length (m)",
    "straightness": "Straightness", "mean_speed_m_s": "Mean speed (m/s)",
    "max_speed_m_s": "Max speed (m/s)", "n_distinct_classes": "# distinct classes",
    "n_fragments": "# fragments", "start_distance_m_base": "Start distance (m)",
    "end_distance_m_base": "End distance (m)",
    "closest_approach_m_base": "Closest approach (m)",
    "is_fragmented": "Is fragmented", "is_stationary": "Is stationary",
    "side_pass": "Side pass", "crosses_ego_path": "Crosses ego path",
}


def _coerce_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _eval_condition(tracklet: dict, cond: dict) -> bool:
    """Evaluate one {field, op, value} condition against a tracklet."""
    field = cond.get("field")
    op = cond.get("op")
    raw = cond.get("value")
    if not field or not op:
        return False
    actual = tracklet.get(field)

    if op in ("<", "<=", ">", ">="):
        a, b = _coerce_num(actual), _coerce_num(raw)
        if a is None or b is None:
            return False
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

    if op in ("in", "not in"):
        vals = [v.strip() for v in str(raw).split(",") if v.strip()]
        member = str(actual) in vals
        return member if op == "in" else not member

    # "==" / "!=": bool-aware, then numeric-aware, else string compare.
    if isinstance(actual, bool):
        b = _coerce_bool(raw)
        eq = (b is not None and actual == b)
    else:
        an, bn = _coerce_num(actual), _coerce_num(raw)
        eq = (an == bn) if (an is not None and bn is not None) else (str(actual) == str(raw))
    return eq if op == "==" else not eq


def _eval_tracklet(tracklet: dict, conditions: list, combinator: str) -> bool:
    results = [_eval_condition(tracklet, c) for c in conditions]
    return all(results) if combinator == "all" else any(results)


def _field_values(ctx) -> tuple[dict, int]:
    """Per-field sorted distinct values for each categorical field (so the
    filter form can suggest only the values valid for the chosen field) plus the
    total tracklet count (``n`` is 0 when nothing is built yet)."""
    store = ctx.store(STORE_NAME)
    meta = store.get("meta") or {}
    vals = {f: set() for f in _FILTER_CAT_FIELDS}
    n = 0
    for s in list(meta.get("scenes", [])):
        for t in (store.get(f"tracklets:{s}") or []):
            n += 1
            for f in _FILTER_CAT_FIELDS:
                v = t.get(f)
                if v is not None and v != "":
                    vals[f].add(str(v))
    return {f: sorted(vs) for f, vs in vals.items()}, n


class FilterTrajectories(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="filter_trajectories",
            label="Filter trajectories",
            # Dynamic so each condition row's Value widget can be typed to the
            # field the user picked (field-specific value suggestions).
            dynamic=True,
            description=(
                "Select built tracklets matching one or more conditions "
                "(QC or scene/data mining). Writes a {scene: [track_idx]} "
                "selection that the Trajectories tab grid reflects."
            ),
        )

    def _condition_object(self, field, field_values):
        """Build one condition's nested Object, typed to the chosen field."""
        field_choices = types.Dropdown()
        for f in _FILTER_FIELDS:
            field_choices.add_choice(f, label=_FIELD_LABELS.get(f, f))

        obj = types.Object()
        obj.enum(
            "field", list(_FILTER_FIELDS), view=field_choices,
            required=False, label="Field",
        )
        if field in _FILTER_CAT_FIELDS:
            op_c = types.Choices()
            for o in ("in", "not in", "==", "!="):
                op_c.add_choice(o, label=o)
            obj.enum("op", op_c.values(), default="in", view=op_c, label="Op")
            vac = types.AutocompleteView(allow_user_input=True)
            for v in field_values.get(field, []):
                vac.add_choice(v, label=v)
            obj.str(
                "value", view=vac, label="Value",
                description="Pick a value (or comma-list for in / not in).",
            )
        elif field in _FILTER_NUM_FIELDS:
            op_c = types.Choices()
            for o in ("<", "<=", ">", ">=", "==", "!="):
                op_c.add_choice(o, label=o)
            obj.enum("op", op_c.values(), default=">", view=op_c, label="Op")
            obj.str("value", label="Value", description="A number.")
        elif field in _FILTER_BOOL_FIELDS:
            yn = types.RadioGroup(orientation="horizontal")
            yn.add_choice("true", label="Yes")
            yn.add_choice("false", label="No")
            obj.enum("value", yn.values(), default="true", view=yn, label="Value")
        # else: no field chosen yet → only the Field dropdown is shown.
        return obj

    def resolve_input(self, ctx):
        inputs = types.Object()
        field_values, n = _field_values(ctx)

        if not n:
            inputs.message(
                "empty", label="Build trajectories first",
                description="No tracklets are in the store yet.",
            )
            return types.Property(
                inputs, view=types.View(label="Filter trajectories")
            )

        combinator = types.Choices()
        combinator.add_choice("all", label="Match ALL conditions (AND)")
        combinator.add_choice("any", label="Match ANY condition (OR)")
        inputs.enum(
            "combinator", combinator.values(), default="all", view=combinator,
            label="Combine", required=True,
        )

        # Render one row per filled condition + a trailing blank "add" row.
        # dynamic=True re-runs this as the user picks a field, so each row's
        # Value widget is typed to that field (categorical → dropdown of THAT
        # field's values; numeric → number; boolean → Yes/No).
        n_filled = 0
        while True:
            c = ctx.params.get(f"condition_{n_filled}")
            if not (isinstance(c, dict) and c.get("field")):
                break
            n_filled += 1

        for i in range(n_filled + 1):
            cur = ctx.params.get(f"condition_{i}") or {}
            obj = self._condition_object(cur.get("field"), field_values)
            label = ("Condition " + str(i + 1)) if i < n_filled else "Add condition"
            inputs.define_property(
                f"condition_{i}", obj, view=types.View(label=label),
            )

        inputs.str(
            "save_as", required=False, label="Save filter as",
            description="Optional: save these conditions under this name for "
                        "reuse (per user, across datasets).",
        )
        return types.Property(inputs, view=types.View(label="Filter trajectories"))

    def execute(self, ctx) -> dict[str, Any]:
        store = ctx.store(STORE_NAME)
        meta = store.get("meta") or {}
        scenes = list(meta.get("scenes", []))

        # Saved-filter replay / the "Apply filter" button pass a raw conditions
        # list directly; the dynamic form passes condition_{i} blocks that we
        # reconstruct into the same {field, op, value} shape.
        raw = ctx.params.get("conditions")
        if isinstance(raw, list):
            conditions = raw
        else:
            conditions = []
            i = 0
            while isinstance(ctx.params.get(f"condition_{i}"), dict):
                c = ctx.params[f"condition_{i}"]
                field, value = c.get("field"), c.get("value")
                if field and value not in (None, ""):
                    conditions.append({
                        "field": field, "op": c.get("op") or "==", "value": value,
                    })
                i += 1
        combinator = ctx.params.get("combinator") or "all"

        # No conditions → clear any active selection (show everything).
        if not conditions:
            store.delete("filter_selection")
            return {"cleared": True, "n_checked": 0, "n_matched": 0}

        selection: dict[str, list] = {}
        n_checked = 0
        n_matched = 0
        for s in scenes:
            rows = store.get(f"tracklets:{s}") or []
            for t in rows:
                n_checked += 1
                if _eval_tracklet(t, conditions, combinator):
                    selection.setdefault(s, []).append(int(t["track_idx"]))
                    n_matched += 1

        summary = {"n_checked": n_checked, "n_matched": n_matched,
                   "by_scene": {s: len(v) for s, v in selection.items()}}
        store.set("filter_selection", {
            "updated": time.time(),
            "spec": {"combinator": combinator, "conditions": conditions},
            "selection": selection,
            "summary": summary,
        })

        save_as = (ctx.params.get("save_as") or "").strip()
        if save_as:
            _saved_filter_store().set(_saved_filter_key(ctx, save_as), {
                "name": save_as,
                "combinator": combinator,
                "conditions": conditions,
                "saved_at": time.time(),
            })
            summary["saved_as"] = save_as
        return summary


# -----------------------------------------------------------------------------
# Saved filters — per-user, reusable across datasets (global store)
# -----------------------------------------------------------------------------

# Filter specs are reusable across datasets, so they live in a GLOBAL store
# (dataset_id=None), namespaced per user. Built tracklets + the live
# selection stay in the dataset-scoped ctx.store(STORE_NAME).
SAVED_FILTERS_STORE = "object_tracking_filters"


def _user_key(ctx) -> str:
    u = getattr(ctx, "user", None)
    if u is not None:
        return str(getattr(u, "id", None) or getattr(u, "email", None) or "anon")
    return "anon"


def _saved_filter_store():
    return foo.ExecutionStore.create(SAVED_FILTERS_STORE, dataset_id=None)


def _saved_filter_key(ctx, name: str) -> str:
    return f"{_user_key(ctx)}:{name}"


# Saved clustering runs — same global, per-user pattern as saved filters.
# Each run stores its full param set so it can be re-run (recalled) later,
# surviving rebuilds (which invalidate the cached cluster blobs).
SAVED_RUNS_STORE = "object_tracking_cluster_runs"


def _saved_run_store():
    return foo.ExecutionStore.create(SAVED_RUNS_STORE, dataset_id=None)


def _saved_run_key(ctx, name: str) -> str:
    return f"{_user_key(ctx)}:{name}"


class ListTrajectoryFilters(foo.Operator):
    """Return the current user's saved filter specs for the tab dropdown."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="list_trajectory_filters",
            label="List trajectory filters",
            unlisted=True,
        )

    def execute(self, ctx) -> dict[str, Any]:
        store = _saved_filter_store()
        prefix = _user_key(ctx) + ":"
        filters = []
        for key in store.list_keys():
            if key.startswith(prefix):
                spec = store.get(key)
                if spec:
                    filters.append(spec)
        filters.sort(key=lambda d: d.get("name", ""))
        return {"filters": filters}


class DeleteTrajectoryFilter(foo.Operator):
    """Delete one of the current user's saved filter specs by name."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="delete_trajectory_filter",
            label="Delete trajectory filter",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("name", required=True)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        name = ctx.params["name"]
        try:
            deleted = _saved_filter_store().delete(_saved_filter_key(ctx, name))
        except Exception:
            # Already gone / never existed — treat as a successful no-op
            # rather than surfacing a "Failed to execute" toast.
            deleted = False
        return {"deleted": bool(deleted), "name": name}


class ClearTrajectoryFilter(foo.Operator):
    """Clear the active trajectory-filter selection (show all tracklets).

    A dedicated operator with no required inputs: the Trajectories tab's
    "Clear filter" button can invoke it directly without the operator prompt,
    avoiding the input-validation failure that ``filter_trajectories`` raised
    when executed with no ``combinator``.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="clear_trajectory_filter",
            label="Clear trajectory filter",
            unlisted=True,
        )

    def execute(self, ctx) -> dict[str, Any]:
        try:
            ctx.store(STORE_NAME).delete("filter_selection")
        except Exception:
            pass
        return {"cleared": True}


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
        # Resolve each filepath into a value the JS layer can turn into a
        # browser-loadable URL:
        #   * HTTP            -> already loadable; pass through
        #   * LOCAL           -> raw path; the frontend routes it through the
        #                        App's /media server via fos.getSampleSrc
        #                        (fos.get_url would *raise* here — local file
        #                        systems don't support signed URLs)
        #   * cloud (GCS/S3/  -> signed URL
        #     Azure/MinIO)
        # Cached client-side per (scene, slice); the JS layer doesn't re-fire
        # unless the user changes either.
        passthrough_fs = (fos.FileSystem.HTTP, fos.FileSystem.LOCAL)
        frame_urls = {}
        for fi, fp in zip(frame_idxs, filepaths):
            if fi is None or not fp:
                continue
            if fos.get_file_system(fp) in passthrough_fs:
                url = fp
            else:
                try:
                    url = fos.get_url(fp)
                except Exception as e:
                    print(f"[get_camera_frame_urls] cannot sign {fp!r}: {e}")
                    continue
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
# tag_trajectories — add / remove tags on selected trajectories
# -----------------------------------------------------------------------------

class TagTrajectories(foo.Operator):
    """Add or remove tags on a set of selected trajectories.

    Tags are written through to the underlying detection *label* tags —
    every Detection sharing the trajectory's ``instance._id`` across every
    group slice that has a ``detections`` field (the lidar cuboids and the
    camera 2D boxes) — so they are durable, filterable in the App sidebar
    wherever the track is viewed, and re-hydrated onto the tracklets by the
    next ``build_trajectories``.
    The built tracklets in the store are also updated in place so the
    Trajectories-tab grid reflects the change immediately.

    Inputs:
      selection: list of {scene_name, instance_id, track_idx} — the
        trajectories to (un)tag. Ego rows (no instance_id) are skipped.
      tags: list[str] — the tags to add or remove.
      mode: "add" (default) or "remove".
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="tag_trajectories",
            label="Tag trajectories",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.list("selection", types.Object(), required=True)
        inputs.list("tags", types.String(), required=True)
        inputs.str("mode", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        from bson import ObjectId

        ds = ctx.dataset
        if ds is None:
            return {"error": "No dataset loaded."}
        lidar_slice = _resolve_lidar_slice(ds.group_slices)
        if lidar_slice is None:
            return {"error": "No lidar group slice found on this dataset."}

        tags = [
            str(t).strip()
            for t in (ctx.params.get("tags") or [])
            if str(t).strip()
        ]
        if not tags:
            return {"error": "No tags given."}
        mode = (ctx.params.get("mode") or "add").lower()
        if mode not in ("add", "remove"):
            return {"error": f"invalid mode {mode!r}"}

        # Group the selection by scene, keeping only real (object) tracks.
        by_scene: dict[str, list[str]] = {}
        for item in (ctx.params.get("selection") or []):
            scene = item.get("scene_name")
            iid = item.get("instance_id")
            if not scene or not iid:
                continue
            by_scene.setdefault(scene, []).append(str(iid))

        # Every group slice carrying a ``detections`` field — the lidar
        # cuboids plus the camera 2D boxes — shares the same instance ids,
        # so the tag lands on the track wherever it's viewed.
        det_slices = _detection_slices(ds)
        if not det_slices:
            return {"error": "No group slice has a 'detections' field."}

        store = ctx.store(STORE_NAME)
        n_tracklets = 0
        n_labels = 0
        by_scene_counts: dict[str, int] = {}

        for scene, hexes in by_scene.items():
            try:
                oids = [ObjectId(h) for h in hexes]
            except Exception as e:
                return {"error": f"invalid instance hex in {hexes!r}: {e!r}"}

            # ----- write-through to the detection label tags (all slices) -----
            for sl in det_slices:
                view = (
                    ds.select_group_slices(sl)
                      .match(F("scene_name") == scene)
                      .filter_labels("detections", F("instance._id").is_in(oids))
                )
                if mode == "add":
                    view.tag_labels(tags, label_fields="detections")
                else:
                    view.untag_labels(tags, label_fields="detections")
                n_labels += int(view.count("detections.detections"))

            # ----- mirror onto the built tracklets in the store -----
            rows = store.get(f"tracklets:{scene}") or []
            want = set(hexes)
            for t in rows:
                if str(t.get("instance_id")) in want:
                    cur = set(t.get("tags") or [])
                    if mode == "add":
                        cur.update(tags)
                    else:
                        cur.difference_update(tags)
                    t["tags"] = sorted(cur)
                    n_tracklets += 1
            store.set(f"tracklets:{scene}", rows)
            by_scene_counts[scene] = len(hexes)

        return {
            "mode": mode,
            "tags": tags,
            "n_tracklets": n_tracklets,
            "n_labels": n_labels,
            "by_scene": by_scene_counts,
        }


# -----------------------------------------------------------------------------
# export_trajectories — download the selected trajectories as JSON
# -----------------------------------------------------------------------------

_EXPORT_IDENT_FIELDS = (
    "scene_name", "tracking_id", "instance_id", "tracking_name", "track_idx",
)


class ExportTrajectories(foo.Operator):
    """Export the selected trajectories as a JSON file (browser download).

    Builds a ``{scene_name: [record, ...]}`` document where each record
    carries identifiers, the scalar metadata, the tags, and — when
    ``include_xy`` — the per-frame base/world XY paths. Delivered via the
    browser's download as a base64 ``data:`` URL, so it works on remote
    FOE deployments with no server-served file.

    Inputs:
      selection: list of {scene_name, track_idx} (instance_id optional).
      include_xy: bool (default True) — include xy_base / xy_world arrays.
      filename: str (optional) — download filename.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="export_trajectories",
            label="Export trajectories",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.list("selection", types.Object(), required=True)
        inputs.bool("include_xy", default=True, required=False)
        inputs.str("filename", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        import base64
        import json

        ds = ctx.dataset
        store = ctx.store(STORE_NAME)
        meta = store.get("meta") or {}
        include_xy = bool(ctx.params.get("include_xy", True))

        # scene -> set(track_idx) requested
        want: dict[str, set] = {}
        for item in (ctx.params.get("selection") or []):
            scene = item.get("scene_name")
            if scene is None:
                continue
            want.setdefault(scene, set()).add(int(item["track_idx"]))

        scenes_out: dict[str, list] = {}
        n = 0
        for scene, idxs in want.items():
            recs = []
            for t in (store.get(f"tracklets:{scene}") or []):
                if int(t["track_idx"]) not in idxs:
                    continue
                rec = {k: t.get(k) for k in _EXPORT_IDENT_FIELDS}
                for k in _TRACKLET_SCALARS:
                    rec[k] = t.get(k)
                rec["tags"] = list(t.get("tags") or [])
                if include_xy:
                    rec["frame_indices"] = t.get("frame_indices")
                    rec["xy_base"] = t.get("xy_base")
                    rec["xy_world"] = t.get("xy_world")
                recs.append(rec)
                n += 1
            if recs:
                scenes_out[scene] = recs

        doc = {
            "source_dataset": meta.get("source") or (ds.name if ds else None),
            "exported_at": time.time(),
            "n_trajectories": n,
            "include_xy": include_xy,
            "scenes": scenes_out,
        }

        payload = json.dumps(doc, indent=2)
        b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        url = f"data:application/json;base64,{b64}"
        filename = (ctx.params.get("filename") or "").strip()
        if not filename:
            src = (meta.get("source") or "trajectories").replace("/", "_")
            filename = f"{src}_trajectories.json"
        ctx.ops.browser_download(url, filename=filename)

        return {
            "n_trajectories": n,
            "n_scenes": len(scenes_out),
            "filename": filename,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("n_trajectories", label="Trajectories exported")
        outputs.int("n_scenes", label="Scenes")
        outputs.str("filename", label="File")
        return types.Property(outputs, view=types.View(label="Export complete"))


# -----------------------------------------------------------------------------
# cluster_trajectories — DTW + hierarchical clustering of trajectory shapes
# -----------------------------------------------------------------------------

# Hard cap on trajectories clustered per scene. DTW is O(N^2 * T^2) with a
# Python outer loop, so an unbounded all-scenes run could stall the worker.
# Above the cap we keep the longest tracks and report n_clustered/n_total so
# the panel can surface "Clustered N of M" — never a silent truncation.
DEFAULT_MAX_CLUSTER_TRACKS = 400

# Arc-length-downsample each path to at most this many points before DTW.
# DTW is O(T_a * T_b) per pair; capping T keeps a several-hundred-track scene
# at seconds rather than minutes, with negligible effect on shape clusters.
# (Measured: a 278-track scene with paths up to 700 points went 350s → 16s.)
DEFAULT_RESAMPLE_POINTS = 30

_CLUSTER_METHODS = ("average", "single", "complete", "ward")
_CLUSTER_FRAMES = ("base", "world", "scene_local")


def _built_classes(store, meta) -> list[str]:
    """Distinct object classes across built tracklets (ego excluded).

    Reads ``meta["classes"]`` (stamped by build_trajectories) when present;
    falls back to scanning tracklets for builds made before that field
    existed.
    """
    cls = (meta or {}).get("classes")
    if cls:
        return sorted(cls)
    seen: set[str] = set()
    for s in (meta or {}).get("scenes", []):
        for t in (store.get(f"tracklets:{s}") or []):
            if t.get("kind") == "ego":
                seen.add("ego")
            elif t.get("tracking_name"):
                seen.add(str(t["tracking_name"]))
    return sorted(seen)


def _cluster_params(ctx) -> dict:
    """Normalized clustering params; also the fingerprint stored per scene."""
    dt = ctx.params.get("distance_threshold")
    try:
        dt = float(dt) if dt not in (None, "") else None
        if dt is not None and dt <= 0:
            dt = None
    except (TypeError, ValueError):
        dt = None
    band = ctx.params.get("band")
    try:
        band = int(band) if band not in (None, "") else None
    except (TypeError, ValueError):
        band = None
    raw_classes = ctx.params.get("classes")
    classes = sorted({str(c) for c in raw_classes}) if raw_classes else None
    return {
        "frame": ctx.params.get("frame") or "world",
        "normalize": bool(ctx.params.get("normalize", True)),
        "method": ctx.params.get("method") or "average",
        "num_clusters": int(ctx.params.get("num_clusters") or 4),
        "distance_threshold": dt,
        "band": band,
        "resample_points": int(ctx.params.get("resample_points") or DEFAULT_RESAMPLE_POINTS),
        "classes": classes,  # None = all object classes
    }


class ClusterTrajectories(foo.Operator):
    """Cluster built trajectories by shape via DTW + hierarchical clustering.

    Reads the tracklets that ``build_trajectories`` put in the store,
    computes a pairwise Dynamic Time Warping distance matrix over each
    scene's object trajectories, runs agglomerative clustering, and
    stores the linkage + cut + dendrogram geometry under
    ``clusters:{scene}`` for the panel's Clusters tab to render.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="cluster_trajectories",
            label="Cluster trajectories",
            description=(
                "Group built trajectories by shape (Dynamic Time Warping + "
                "hierarchical clustering) into a dendrogram the Clusters tab "
                "renders. Optionally restrict to a subset of object classes "
                "(ego is excluded — one track per scene). Clusters are stored "
                "per scene; rebuilding trajectories invalidates them."
            ),
            # DTW is O(N^2 * T^2); a big all-scenes run can be long, so allow
            # scheduling it as a delegated operation (immediate still allowed
            # for small scenes). Delegated runs require an orchestrator on the
            # deployment, else they queue.
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        store = ctx.store(STORE_NAME)
        built = list((store.get("meta") or {}).get("scenes", []))
        if not built:
            inputs.message(
                "empty", label="Build trajectories first",
                description="No tracklets are in the store yet.",
            )
            return types.Property(inputs, view=types.View(label="Cluster trajectories"))

        scene_c = types.Choices()
        scene_c.add_choice("__all__", label="All scenes (pooled)")
        for s in built:
            scene_c.add_choice(s, label=s)
        inputs.enum(
            "scene", scene_c.values(), default=built[0], view=scene_c,
            required=True, label="Scene",
            description="Cluster one scene, or pool the chosen classes across "
                        "ALL scenes into one dendrogram (so e.g. every run's "
                        "ego path clusters together).",
        )

        frame_c = types.Choices()
        frame_c.add_choice("world", label="World frame")
        frame_c.add_choice("base", label="Ego/base frame")
        frame_c.add_choice("scene_local", label="Scene-local")
        inputs.enum(
            "frame", frame_c.values(), default="world", view=frame_c,
            label="Reference frame",
        )
        inputs.bool(
            "normalize", default=True, label="Normalize shape",
            description="Translate each path to the origin and rotate so its "
                        "start→end chord points the same way, so clustering "
                        "groups by shape regardless of position/heading.",
        )

        classes = _built_classes(ctx.store(STORE_NAME), store.get("meta") or {})
        if classes:
            class_c = types.Choices()
            for c in classes:
                class_c.add_choice(c, label=c)
            inputs.list(
                "classes", types.String(), view=class_c, default=list(classes),
                required=False, label="Classes",
                description="Restrict clustering to these object classes "
                            "(default: all). Fewer classes = fewer, clearer "
                            "clusters and faster compute.",
            )

        method_c = types.Choices()
        for m in _CLUSTER_METHODS:
            method_c.add_choice(m, label=m)
        inputs.enum(
            "method", method_c.values(), default="average", view=method_c,
            label="Linkage method",
        )
        inputs.int(
            "num_clusters", default=4, label="Number of clusters",
            description="Target cluster count (cut by maxclust).",
        )
        inputs.float(
            "distance_threshold", required=False,
            label="Or cut at distance",
            description="If set (> 0), cut the tree at this DTW distance "
                        "instead of by count.",
        )
        inputs.int(
            "resample_points", default=DEFAULT_RESAMPLE_POINTS,
            label="Resample points (advanced)",
            description="Arc-length-downsample each path to at most this many "
                        "points before DTW (bounds compute; 0 = full length).",
        )
        inputs.int(
            "band", required=False, label="DTW band (advanced)",
            description="Optional Sakoe-Chiba radius to speed up DTW.",
        )
        inputs.int(
            "max_tracks", default=DEFAULT_MAX_CLUSTER_TRACKS,
            label="Max trajectories (advanced)",
            description="Cap on the clustered set; the longest tracks are kept.",
        )
        inputs.str(
            "save_as", required=False, label="Save run as",
            description="Optional: save this configuration under a name to "
                        "recall it later (per user, survives rebuilds).",
        )
        return types.Property(inputs, view=types.View(label="Cluster trajectories"))

    def execute(self, ctx) -> dict[str, Any]:
        store = ctx.store(STORE_NAME)
        meta = store.get("meta") or {}
        built = list(meta.get("scenes", []))
        if not built:
            return {"error": "No tracklets built. Run build_trajectories first."}

        scene_param = ctx.params.get("scene") or built[0]
        pooled = scene_param == "__all__"
        params = _cluster_params(ctx)
        max_tracks = int(ctx.params.get("max_tracks") or DEFAULT_MAX_CLUSTER_TRACKS)
        allow = set(params["classes"]) if params["classes"] is not None else None

        # (scene, tracklet) pairs passing the class filter. Ego counts as class
        # "ego"; with no class filter (default) ego is excluded — one per scene
        # is not clusterable on its own, so it's opt-in via the class picker.
        def _gather(scene_list):
            out = []
            for scene in scene_list:
                for t in (store.get(f"tracklets:{scene}") or []):
                    is_ego = t.get("kind") == "ego"
                    ck = "ego" if is_ego else t.get("tracking_name")
                    if allow is None:
                        if is_ego:
                            continue
                    elif ck not in allow:
                        continue
                    out.append((scene, t))
            return out

        def _cluster(result_key, scene_list, scope):
            members, arrays, skipped = tracks_to_arrays(
                _gather(scene_list), frame_key=params["frame"],
                normalize=params["normalize"], resample=params["resample_points"],
            )
            n_total = len(members)
            capped = False
            if n_total > max_tracks:
                order = sorted(range(n_total), key=lambda i: -arrays[i].shape[0])
                keep = sorted(order[:max_tracks])
                members = [members[i] for i in keep]
                arrays = [arrays[i] for i in keep]
                capped = True
            n_clustered = len(members)

            base = {
                "scene_name": result_key, "scope": scope,
                "frame": params["frame"], "normalize": params["normalize"],
                "classes": params["classes"],
                "n_total": n_total, "n_clustered": n_clustered,
                "capped": capped, "max_tracks": max_tracks,
                "skipped": skipped, "params": params, "updated": time.time(),
            }
            if n_clustered < 2:
                base["error"] = ("Need at least 2 trajectories to cluster "
                                 "(try more classes, or All scenes).")
                store.set(f"clusters:{result_key}", base)
                return None

            # Serial by design: joblib subprocesses are unreliable inside an
            # operator worker; the max_tracks cap + resampling bound the work.
            dist = pairwise_dtw_matrix(arrays, band=params["band"], n_jobs=1)
            warnings: list[str] = []
            if not np.all(np.isfinite(dist)):
                finite = dist[np.isfinite(dist)]
                fill = float(finite.max()) if finite.size else 1.0
                dist = np.where(np.isfinite(dist), dist, fill)
                np.fill_diagonal(dist, 0.0)
                warnings.append("non-finite distances clamped")

            result = cluster_from_matrix(
                dist, method=params["method"],
                num_clusters=(None if params["distance_threshold"] else params["num_clusters"]),
                distance_threshold=params["distance_threshold"],
            )
            result.update(base)
            result["members"] = members  # row → {scene_name, track_idx}
            result["warnings"] = warnings
            store.set(f"clusters:{result_key}", result)
            return result_key

        # Pool every built scene into one clustering for "All scenes" (so e.g.
        # every run's ego path clusters together); otherwise just the one scene.
        if pooled:
            done = [_cluster("__all__", built, "all")]
        else:
            done = [_cluster(scene_param, [scene_param], "scene")]
        clustered = [k for k in done if k]

        # Accumulate the set of currently-clustered keys so the panel's scene
        # picker can browse every run made since the last build (rebuild clears
        # this). Keys are scene names or "__all__", so union-by-key is safe.
        prev = list((store.get("clusters_meta") or {}).get("scenes", []))
        store.set("clusters_meta", {
            "updated": time.time(),
            "scenes": sorted(set(prev) | set(clustered)),
            "params": params,
        })

        saved = None
        save_as = (ctx.params.get("save_as") or "").strip()
        if save_as:
            _saved_run_store().set(_saved_run_key(ctx, save_as), {
                "name": save_as,
                "scene": scene_param,
                "params": params,
                "max_tracks": max_tracks,
                "saved_at": time.time(),
            })
            saved = save_as

        return {"scenes": clustered, "n_scenes": len(clustered), "saved_as": saved}


class GetClusters(foo.Operator):
    """Read stored cluster results for the Clusters-tab dendrogram.

    Cheap read path (mirrors ``get_trajectories``) so switching scenes or
    reopening the tab never re-runs the O(N^2) DTW compute.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_clusters",
            label="Get clusters",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("scene_name", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        store = ctx.store(STORE_NAME)
        cmeta = store.get("clusters_meta") or {}
        scene = ctx.params.get("scene_name")
        scenes = [scene] if scene else list(cmeta.get("scenes", []))
        clusters = {}
        for s in scenes:
            blob = store.get(f"clusters:{s}")
            if blob is not None:
                clusters[s] = blob
        return {
            "scene_name": scene,
            "clusters": clusters,
            "clusters_meta": cmeta,
        }


class SelectTrajectories(foo.Operator):
    """Write a ``{scene: [track_idx, ...]}`` selection into the store.

    The seam that converges any selection source (a clicked cluster, a
    future similarity search) onto the same ``filter_selection`` the
    Trajectories grid and BEV view already read — so highlighting needs no
    new code. Pass an empty/missing selection to clear it.

    ``selection`` is a list of ``{scene_name, track_idx}`` (same shape as
    ``tag_trajectories`` / ``export_trajectories``); a declared
    ``resolve_input`` is REQUIRED so the executor passes the param through
    — an operator with no input schema has its params dropped on FiftyOne
    ≥ 2.18, which silently clears the selection instead of setting it.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="select_trajectories",
            label="Select trajectories",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.list("selection", types.Object(), required=True)
        inputs.str("source", required=False)
        inputs.int("cluster_id", required=False)
        inputs.str("label", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        store = ctx.store(STORE_NAME)
        clean: dict[str, list] = {}
        total = 0
        for item in (ctx.params.get("selection") or []):
            scene = item.get("scene_name")
            if scene is None or item.get("track_idx") is None:
                continue
            clean.setdefault(scene, []).append(int(item["track_idx"]))
            total += 1

        if not clean:
            store.delete("filter_selection")
            return {"cleared": True, "n_matched": 0}

        summary = {
            "n_matched": total,
            "by_scene": {s: len(v) for s, v in clean.items()},
        }
        label = ctx.params.get("label")
        if label:
            summary["label"] = label
        store.set("filter_selection", {
            "updated": time.time(),
            "spec": {
                "source": ctx.params.get("source") or "select",
                "cluster_id": ctx.params.get("cluster_id"),
            },
            "selection": clean,
            "summary": summary,
        })
        return summary


class ListClusterRuns(foo.Operator):
    """Return the current user's saved clustering runs for the tab dropdown."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="list_cluster_runs", label="List cluster runs", unlisted=True,
        )

    def execute(self, ctx) -> dict[str, Any]:
        store = _saved_run_store()
        prefix = _user_key(ctx) + ":"
        runs = []
        for key in store.list_keys():
            if key.startswith(prefix):
                spec = store.get(key)
                if spec:
                    runs.append(spec)
        runs.sort(key=lambda d: d.get("name", ""))
        return {"runs": runs}


class DeleteClusterRun(foo.Operator):
    """Delete one saved clustering run by name (current user)."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="delete_cluster_run", label="Delete cluster run", unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("name", required=True)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        name = (ctx.params.get("name") or "").strip()
        if not name:
            return {"error": "No run name given."}
        _saved_run_store().delete(_saved_run_key(ctx, name))
        return {"deleted": name}


# -----------------------------------------------------------------------------
# Plugin registration
# -----------------------------------------------------------------------------

def register(p):
    p.register(ListTrackingScenes)
    p.register(ResolveSceneForSample)
    p.register(GetSceneTrackPayload)
    p.register(GetCameraFrameUrls)
    p.register(ViewTrackPatches)
    p.register(BuildTrajectories)
    p.register(GetTrajectories)
    p.register(FilterTrajectories)
    p.register(ClearTrajectoryFilter)
    p.register(ListTrajectoryFilters)
    p.register(DeleteTrajectoryFilter)
    p.register(TagTrajectories)
    p.register(ExportTrajectories)
    p.register(ClusterTrajectories)
    p.register(GetClusters)
    p.register(SelectTrajectories)
    p.register(ListClusterRuns)
    p.register(DeleteClusterRun)
