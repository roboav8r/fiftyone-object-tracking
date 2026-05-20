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
    """Switch the App view to one-patch-per-frame for a single track.

    Flattens the dataset across one or more camera slices, filters to
    a single scene + ``fo.Instance.id`` (the FO-side cross-frame
    identifier shared by every Detection of the same track), and
    applies the resulting patches view via ``ctx.ops.set_view``.

    Why ``instance._id`` instead of the source-side ``tracking_id``?
    The source-side field name is dataset-specific (``tracking_id``
    for our loaders, but e.g. KITTI uses ``track_id``, public MOT
    benchmarks use other names) AND can be stored as different types
    (str vs int). The FO ``Instance`` id is dataset-agnostic and the
    same for all of a track's Detections regardless of loader. JS
    already has the selected track's instance hex (it's how the BEV
    panel highlights selected tracks), so the operator stays clean.

    Advanced users can override via ``match_field`` + ``match_value``
    to filter by any other Detection-level field (e.g.
    ``"detections.tracking_id"`` or a custom dynamic field).

    Inputs:
      scene_name: str — required.
      instance_id: str — required FO ``fo.Instance`` hex; the
        ``ObjectId`` form of ``detection.instance._id``. JS passes
        this directly from the panel's ``selectedInstanceId`` state.
      camera_slices: list[str] (or comma-separated string) —
        optional. When omitted/empty, the operator auto-picks every
        non-lidar group slice (i.e. all cameras). ``camera_slice``
        (singular str) is also accepted as a single-slice shortcut.
      match_field, match_value: optional advanced override. When set,
        replaces the ``detections.instance._id`` match with
        ``F(match_field) == match_value`` (or its int form if it
        parses). Useful for filtering by ``detections.tracking_id``
        or any other Detection-level field.
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
        inputs.str("instance_id", required=False)
        # Multi-camera surface: either a list (camera_slices) or a
        # single name (camera_slice — legacy). When neither is set,
        # the operator picks every non-lidar slice.
        inputs.list("camera_slices", types.String(), required=False)
        inputs.str("camera_slice", required=False)
        # Advanced overrides
        inputs.str("match_field", required=False)
        inputs.str("match_value", required=False)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        from bson import ObjectId

        scene_name = ctx.params["scene_name"]
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
            # Auto-pick: every slice whose name doesn't start with
            # "lidar". Captures all cameras across our loaders
            # (cam_*, image_0*, front*, back_*, *_distorted, …) and
            # skips lidar / lidar_livox / lidar_hesai.
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
        instance_id = ctx.params.get("instance_id")

        if match_field:
            if match_value is None:
                return {"error": "match_field set but match_value missing"}
            # Try the literal string; also coerce to int if it parses
            try:
                tid_int = int(match_value)
                identity_filter = (
                    (F(match_field) == match_value)
                    | (F(match_field) == tid_int)
                )
            except (ValueError, TypeError):
                identity_filter = F(match_field) == match_value
            identity_desc = f"{match_field}={match_value!r}"
        elif instance_id:
            try:
                oid = ObjectId(instance_id)
            except Exception as e:
                return {
                    "error": f"instance_id {instance_id!r} is not a valid ObjectId: {e!r}"
                }
            identity_filter = F("detections.instance._id") == oid
            identity_desc = f"instance._id={instance_id}"
        else:
            return {
                "error": "either instance_id or (match_field + match_value) is required"
            }

        # to_patches('detections') promotes each Detection embedded
        # doc to a top-level "detections" field on the patches sample
        # — so per-Detection fields live at "detections.<name>" on
        # the patches, not at the patch root.
        view = (
            ctx.dataset.select_group_slices(slices)
                .match(F("scene_name") == scene_name)
                .to_patches("detections")
                .match(identity_filter)
                .sort_by("sample_id")
        )
        n = view.count()
        if n == 0:
            return {
                "error": (
                    f"no patches for {identity_desc} on slices "
                    f"{slices!r} in scene {scene_name!r}"
                )
            }

        ctx.ops.set_view(view=view)
        return {
            "scene_name": scene_name,
            "identity": identity_desc,
            "camera_slices": slices,
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
