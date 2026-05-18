"""Operators for the FiftyOne object-tracking toolkit plugin.

Four operators, exposed through ``register()``:

  - ``list_tracking_scenes``    — enumerate scenes + group-slice names
                                  (consumed by the BEV panel)
  - ``get_scene_track_payload`` — one-shot per-scene trajectory bundle
                                  (consumed by ``BEVTrackVisualization``)
  - ``read_trajectory_payload`` — per-sample parquet → JSON columns
                                  (consumed by ``TrajectoryRenderer``)
  - ``build_trajectories``      — build the per-trajectory FiftyOne
                                  dataset from a grouped tracking
                                  dataset. Listed; surfaces a form
                                  in the App's operator browser.

The first three are unlisted utilities for the JS UI. ``build_trajectories``
is the only user-invoked operator; it takes a source dataset name, a
target dataset name, a ``trajectory_root`` (where per-trajectory
Parquet files land), and an ``overwrite`` flag.
"""

from __future__ import annotations

import functools
import io
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
    PARQUET_COLUMNS,
    TrajectoryRecord,
    build_track_records,
    write_trajectory_parquet,
)
from ._schema import (
    SAMPLE_SCHEMA,
    declare_schema,
    set_sidebar_groups,
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
# 3. read_trajectory_payload
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _read_parquet_columns(filepath: str) -> dict[str, Any]:
    """Read a trajectory parquet file and return the columns the JS
    renderer needs. Cached so a scrolling grid only pays the I/O once."""
    import pyarrow.parquet as pq

    data = fos.read_file(filepath, binary=True)
    table = pq.read_table(io.BytesIO(data))
    raw_md = table.schema.metadata or {}
    md = {
        (k.decode() if isinstance(k, bytes) else k):
        (v.decode() if isinstance(v, bytes) else v)
        for k, v in raw_md.items()
    }

    def col(name: str) -> list:
        if name not in table.column_names:
            return []
        return table.column(name).to_pylist()

    return {
        "color_hex": md.get("color_hex", "#cccccc"),
        "schema_version": md.get("schema_version", "0"),
        "x_base": col("x_base"),
        "y_base": col("y_base"),
        "x_scene_local": col("x_scene_local"),
        "y_scene_local": col("y_scene_local"),
        "fragment_ids": col("fragment_id"),
    }


class ReadTrajectoryPayload(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="read_trajectory_payload",
            label="Read trajectory payload",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sample_id", required=True)
        return types.Property(inputs)

    def execute(self, ctx) -> dict[str, Any]:
        sample_id = ctx.params["sample_id"]
        try:
            sample = ctx.dataset[sample_id]
        except KeyError:
            return {"error": f"Sample {sample_id!r} not found."}

        filepath = sample.filepath
        if not filepath or not filepath.endswith(".parquet"):
            return {
                "error": f"Sample filepath is not a parquet file: {filepath!r}",
            }

        ego_size = list(
            (ctx.dataset.info or {}).get("ego_size_lwh_m")
            or DEFAULT_EGO_SIZE_LWH
        )

        try:
            payload = dict(_read_parquet_columns(filepath))
        except Exception as e:
            return {"error": f"Failed to read {filepath}: {e!r}"}

        payload["ego_size_lwh_m"] = ego_size
        return payload


# -----------------------------------------------------------------------------
# 4. build_trajectories — the user-facing operator
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
    """Core build logic; the operator wraps this."""
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
        "parquet_columns": PARQUET_COLUMNS,
        "parquet_schema_version": "2",
    }
    tgt.save()

    tmpdir = tempfile.mkdtemp(prefix=f"trajectories-{target}-")
    print(f"[build_trajectories] writing Parquet files to {tmpdir}")

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
            local_path = os.path.join(tmpdir, record.parquet_basename)
            write_trajectory_parquet(record, local_path)
            samples.append(_record_to_sample(record, local_path))

    print(f"[build_trajectories] adding {len(samples)} samples to {target}")
    tgt.add_samples(samples)
    tgt.compute_metadata()

    remote_dir = f"{trajectory_root.rstrip('/')}/{target}"
    print(f"[build_trajectories] copying Parquet files to {remote_dir}")
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
# Plugin registration
# -----------------------------------------------------------------------------

def register(p):
    p.register(ListTrackingScenes)
    p.register(GetSceneTrackPayload)
    p.register(ReadTrajectoryPayload)
    p.register(BuildTrajectories)
