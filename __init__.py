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
            store.set(f"tracklets:{scene_name}", tracklets)
            built.append(scene_name)
            total += len(tracklets)

        # A new build invalidates any prior filter selection.
        store.delete("filter_selection")
        store.set("meta", {
            "updated": time.time(),
            "source": ds.name,
            "scenes": built,
            "ego_size_lwh_m": list(ego_size),
            "n_trajectories": total,
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
# Plugin registration
# -----------------------------------------------------------------------------

def register(p):
    p.register(ListTrackingScenes)
    p.register(GetSceneTrackPayload)
    p.register(GetCameraFrameUrls)
    p.register(ViewTrackPatches)
    p.register(BuildTrajectories)
    p.register(GetTrajectories)
