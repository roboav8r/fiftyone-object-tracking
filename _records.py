"""Per-trajectory record builder used by the build operator.

Consumes a per-frame canonical grouped-tracking dataset (cuboids on the
lidar slice, ``world_to_base`` SE(3) per frame, source-side identity
stamped on each Detection) and emits one ``TrajectoryRecord`` per
(scene, FO instance) plus one ego record per scene.

Per-frame arrays are kept in-memory on the ``TrajectoryRecord``; the
build operator hands the record to ``_thumbnail.render_trajectory_thumbnail``
which uses matplotlib to write a PNG to disk. No PyArrow / Parquet
dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ._math import (
    apply_se3,
    gap_stats,
    origin_normalize,
    per_frame_fragment_ids,
    quadrant_base,
    quat_to_yaw,
    scalars_from_xy,
    step_speeds,
    step_velocities,
    timestamps_to_seconds,
)


# Default ego footprint (OSDaR23-style rail vehicle). Each dataset
# writes its actual ego size to ``info["ego_size_lwh_m"]``; the build
# operator reads that and passes it here.
DEFAULT_EGO_SIZE_LWH = (24.0, 2.9, 4.0)


@dataclass
class TrajectoryRecord:
    """One trajectory's data.

    Per-frame arrays (``translations_*``, ``rotations_*``, ``sizes``,
    ``velocities_world``, ``tracking_scores``, ``num_pts``,
    ``fragment_ids``) are kept in-memory and consumed by
    ``_thumbnail.render_trajectory_thumbnail`` at build time. All
    other fields are sample-level scalars surfaced via the FiftyOne
    App sidebar.
    """

    # identity / back-links
    kind: str                # "object" or "ego"
    instance_id: str         # FO instance.id (stable cross-frame link)
    tracking_id: str         # source-side id (dx3 int / raillabel UUID; "ego" for ego)
    segment_index: int
    class_run_idx: int       # vestigial (always 0 post-class-split revert)
    tracking_name: str
    scene_name: str
    source_dataset: str
    sample_token_first: str
    sample_token_last: str

    # coverage scalars
    n_frames: int
    duration_s: float
    frame_idx_first: int
    frame_idx_last: int
    is_fragmented: bool
    n_fragments: int
    n_gap_frames: int
    max_gap_length: int
    is_stationary: bool

    # base-frame position scalars
    start_x_base: float
    start_y_base: float
    end_x_base: float
    end_y_base: float
    start_distance_m_base: float
    end_distance_m_base: float
    start_quadrant_base: str
    end_quadrant_base: str
    closest_approach_m_base: float
    closest_approach_frame_idx: int

    # world position scalars
    start_x_world: float
    start_y_world: float
    end_x_world: float
    end_y_world: float
    displacement_m: float
    path_length_m: float

    # motion scalars
    mean_speed_m_s: float
    max_speed_m_s: float
    min_speed_m_s: float
    speed_std_m_s: float

    # shape scalars
    heading_change_deg: float
    heading_class: str
    straightness: float
    side_pass: bool
    crosses_ego_path: bool

    # QC (annotation-error surfacing)
    n_distinct_classes: int
    tracking_names_distinct: list[str]
    max_step_jump_m: float
    max_gap_s: float

    # AABBs flattened to scalars
    bbox_base_x_min: float
    bbox_base_y_min: float
    bbox_base_x_max: float
    bbox_base_y_max: float
    bbox_world_x_min: float
    bbox_world_y_min: float
    bbox_world_x_max: float
    bbox_world_y_max: float

    # per-frame arrays — kept in-memory; matplotlib renders the BEV
    # thumbnail at build time, no external trajectory file written.
    frame_indices: np.ndarray
    timestamps_ns: list[Optional[str]]
    timestamps_s: np.ndarray
    sample_tokens: list[str]
    translations_base: np.ndarray
    translations_world: np.ndarray
    translations_scene_local: np.ndarray
    translations_origin_normalized: np.ndarray
    rotations_base: np.ndarray
    rotations_world: np.ndarray
    sizes: np.ndarray
    velocities_world: np.ndarray
    tracking_scores: np.ndarray
    num_pts: np.ndarray
    fragment_ids: np.ndarray
    # Filename stem (no extension) for the build operator's
    # per-trajectory PNG. The operator appends ``.png`` and writes
    # the file via matplotlib.
    output_stem: str


# -----------------------------------------------------------------------------
# Record builders
# -----------------------------------------------------------------------------

def _record_from_object(
    *, scene_name, source_dataset, instance_id, tracking_name,
    tracking_id: str = "", segment_index: int = 0, class_run_idx: int = 0,
    tracking_names_distinct: Optional[list[str]] = None,
    frame_indices, sample_tokens, timestamps_ns,
    locations_base, rotations_xyz_base, dimensions,
    wtb_matrices, wtb_quats_world,
) -> TrajectoryRecord:
    order = np.argsort(frame_indices)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)[order]
    sample_tokens = [sample_tokens[i] for i in order]
    timestamps_ns = [timestamps_ns[i] for i in order]
    locations_base = np.asarray(locations_base, dtype=np.float64)[order]
    # rotations_xyz_base = per-frame intrinsic XYZ Euler [rx, ry, rz]
    # (matches fo.Detection.rotation).
    rxyz = np.asarray(rotations_xyz_base, dtype=np.float64)[order]
    dimensions = np.asarray(dimensions, dtype=np.float64)[order]
    wtb_matrices = [wtb_matrices[i] for i in order]
    wtb_quats_world = np.asarray(wtb_quats_world, dtype=np.float64)[order]

    translations_world = np.vstack([
        apply_se3(M, loc.reshape(1, 3))
        for M, loc in zip(wtb_matrices, locations_base)
    ])
    translations_base_3d = locations_base
    translations_scene_local = translations_world - translations_world[0:1]
    translations_origin_normalized_2d = origin_normalize(translations_world[:, :2])

    R_obj_base = R.from_euler("XYZ", rxyz)
    rotations_base = R_obj_base.as_quat()
    R_world_base = R.from_quat(wtb_quats_world)
    rotations_world = (R_world_base * R_obj_base).as_quat()

    t_s = timestamps_to_seconds(timestamps_ns)
    velocities_world = step_velocities(translations_world[:, :2], t_s)
    speeds = step_speeds(translations_world[:, :2], t_s)
    fragment_ids = per_frame_fragment_ids(t_s)
    n_fragments = int(fragment_ids[-1] + 1) if fragment_ids.size else 0
    duration_s = float(t_s[-1] - t_s[0]) if len(t_s) >= 2 else 0.0
    n_gap, max_gap = gap_stats(t_s)

    xy_base = translations_base_3d[:, :2]
    xy_world = translations_world[:, :2]
    sc = scalars_from_xy(xy_base, xy_world, speeds)

    # QC: biggest world-frame XY jump between consecutive keyframes (m).
    if xy_world.shape[0] >= 2:
        max_step_jump_m = float(np.linalg.norm(np.diff(xy_world, axis=0), axis=1).max())
    else:
        max_step_jump_m = 0.0
    # QC: longest inter-keyframe gap in seconds.
    max_gap_s = float(np.diff(t_s).max()) if t_s.shape[0] >= 2 else 0.0

    classes_distinct = (
        sorted(set(tracking_names_distinct))
        if tracking_names_distinct else [tracking_name]
    )

    return TrajectoryRecord(
        kind="object",
        instance_id=instance_id,
        tracking_id=str(tracking_id) if tracking_id else "",
        segment_index=int(segment_index),
        class_run_idx=int(class_run_idx),
        tracking_name=tracking_name,
        scene_name=scene_name,
        source_dataset=source_dataset,
        sample_token_first=sample_tokens[0],
        sample_token_last=sample_tokens[-1],

        n_frames=int(frame_indices.shape[0]),
        duration_s=duration_s,
        frame_idx_first=int(frame_indices[0]),
        frame_idx_last=int(frame_indices[-1]),
        is_fragmented=n_fragments > 1,
        n_fragments=n_fragments,
        n_gap_frames=n_gap,
        max_gap_length=max_gap,
        is_stationary=sc["is_stationary"],

        start_x_base=float(xy_base[0, 0]),
        start_y_base=float(xy_base[0, 1]),
        end_x_base=float(xy_base[-1, 0]),
        end_y_base=float(xy_base[-1, 1]),
        start_distance_m_base=sc["start_distance_m_base"],
        end_distance_m_base=sc["end_distance_m_base"],
        start_quadrant_base=quadrant_base(float(xy_base[0, 0]), float(xy_base[0, 1])),
        end_quadrant_base=quadrant_base(float(xy_base[-1, 0]), float(xy_base[-1, 1])),
        closest_approach_m_base=sc["closest_approach_m_base"],
        closest_approach_frame_idx=int(frame_indices[sc["closest_approach_frame_offset"]]),

        start_x_world=float(xy_world[0, 0]),
        start_y_world=float(xy_world[0, 1]),
        end_x_world=float(xy_world[-1, 0]),
        end_y_world=float(xy_world[-1, 1]),
        displacement_m=sc["displacement_m"],
        path_length_m=sc["path_length_m"],

        mean_speed_m_s=sc["mean_speed_m_s"],
        max_speed_m_s=sc["max_speed_m_s"],
        min_speed_m_s=sc["min_speed_m_s"],
        speed_std_m_s=sc["speed_std_m_s"],

        heading_change_deg=sc["heading_change_deg"],
        heading_class=sc["heading_class"],
        straightness=sc["straightness"],
        side_pass=sc["side_pass"],
        crosses_ego_path=sc["crosses_ego_path"],

        n_distinct_classes=len(classes_distinct),
        tracking_names_distinct=classes_distinct,
        max_step_jump_m=max_step_jump_m,
        max_gap_s=max_gap_s,

        bbox_base_x_min=float(xy_base[:, 0].min()),
        bbox_base_y_min=float(xy_base[:, 1].min()),
        bbox_base_x_max=float(xy_base[:, 0].max()),
        bbox_base_y_max=float(xy_base[:, 1].max()),
        bbox_world_x_min=float(xy_world[:, 0].min()),
        bbox_world_y_min=float(xy_world[:, 1].min()),
        bbox_world_x_max=float(xy_world[:, 0].max()),
        bbox_world_y_max=float(xy_world[:, 1].max()),

        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        timestamps_s=t_s,
        sample_tokens=sample_tokens,
        translations_base=translations_base_3d,
        translations_world=translations_world,
        translations_scene_local=translations_scene_local,
        translations_origin_normalized=translations_origin_normalized_2d,
        rotations_base=rotations_base,
        rotations_world=rotations_world,
        sizes=dimensions,
        velocities_world=velocities_world,
        tracking_scores=np.full(frame_indices.shape[0], -1.0, dtype=np.float64),
        num_pts=np.full(frame_indices.shape[0], -1, dtype=np.int64),
        fragment_ids=fragment_ids,
        output_stem=f"{scene_name}__object__{instance_id}",
    )


def _record_from_ego(
    *, scene_name, source_dataset,
    frame_indices, sample_tokens, timestamps_ns,
    wtb_translations, wtb_quats_xyzw,
    ego_size_lwh: tuple = DEFAULT_EGO_SIZE_LWH,
) -> Optional[TrajectoryRecord]:
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    sample_tokens = [sample_tokens[i] for i in order]
    timestamps_ns = [timestamps_ns[i] for i in order]
    wtb_translations = np.asarray(wtb_translations, dtype=np.float64)[order]
    wtb_quats_xyzw = np.asarray(wtb_quats_xyzw, dtype=np.float64)[order]
    if len(frame_indices) < 2:
        return None

    translations_world = wtb_translations
    translations_base = np.zeros_like(translations_world)
    translations_scene_local = translations_world - translations_world[0:1]
    translations_origin_normalized_2d = origin_normalize(translations_world[:, :2])

    rotations_base = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(frame_indices), 1))
    rotations_world = wtb_quats_xyzw
    sizes = np.tile(np.asarray(ego_size_lwh, dtype=np.float64), (len(frame_indices), 1))

    t_s = timestamps_to_seconds(timestamps_ns)
    velocities_world = step_velocities(translations_world[:, :2], t_s)
    speeds = step_speeds(translations_world[:, :2], t_s)
    fragment_ids = per_frame_fragment_ids(t_s)
    n_fragments = int(fragment_ids[-1] + 1) if fragment_ids.size else 0
    duration_s = float(t_s[-1] - t_s[0]) if len(t_s) >= 2 else 0.0
    n_gap, max_gap = gap_stats(t_s)

    xy_base = translations_base[:, :2]
    xy_world = translations_world[:, :2]
    sc = scalars_from_xy(xy_base, xy_world, speeds)

    max_step_jump_m = (
        float(np.linalg.norm(np.diff(xy_world, axis=0), axis=1).max())
        if xy_world.shape[0] >= 2 else 0.0
    )
    max_gap_s = float(np.diff(t_s).max()) if t_s.shape[0] >= 2 else 0.0

    return TrajectoryRecord(
        kind="ego",
        instance_id="ego",
        tracking_id="ego",
        segment_index=0,
        class_run_idx=0,
        tracking_name="ego",
        scene_name=scene_name,
        source_dataset=source_dataset,
        sample_token_first=sample_tokens[0],
        sample_token_last=sample_tokens[-1],

        n_frames=int(frame_indices.shape[0]),
        duration_s=duration_s,
        frame_idx_first=int(frame_indices[0]),
        frame_idx_last=int(frame_indices[-1]),
        is_fragmented=n_fragments > 1,
        n_fragments=n_fragments,
        n_gap_frames=n_gap,
        max_gap_length=max_gap,
        is_stationary=sc["is_stationary"],

        # Ego is the origin of its own base frame; start/end values are 0.
        start_x_base=0.0, start_y_base=0.0,
        end_x_base=0.0, end_y_base=0.0,
        start_distance_m_base=0.0, end_distance_m_base=0.0,
        start_quadrant_base="front_left", end_quadrant_base="front_left",
        closest_approach_m_base=0.0,
        closest_approach_frame_idx=int(frame_indices[0]),

        start_x_world=float(xy_world[0, 0]),
        start_y_world=float(xy_world[0, 1]),
        end_x_world=float(xy_world[-1, 0]),
        end_y_world=float(xy_world[-1, 1]),
        displacement_m=sc["displacement_m"],
        path_length_m=sc["path_length_m"],

        mean_speed_m_s=sc["mean_speed_m_s"],
        max_speed_m_s=sc["max_speed_m_s"],
        min_speed_m_s=sc["min_speed_m_s"],
        speed_std_m_s=sc["speed_std_m_s"],

        heading_change_deg=sc["heading_change_deg"],
        heading_class=sc["heading_class"],
        straightness=sc["straightness"],
        side_pass=False,
        crosses_ego_path=False,

        n_distinct_classes=1,
        tracking_names_distinct=["ego"],
        max_step_jump_m=max_step_jump_m,
        max_gap_s=max_gap_s,

        bbox_base_x_min=0.0, bbox_base_y_min=0.0,
        bbox_base_x_max=0.0, bbox_base_y_max=0.0,
        bbox_world_x_min=float(xy_world[:, 0].min()),
        bbox_world_y_min=float(xy_world[:, 1].min()),
        bbox_world_x_max=float(xy_world[:, 0].max()),
        bbox_world_y_max=float(xy_world[:, 1].max()),

        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        timestamps_s=t_s,
        sample_tokens=sample_tokens,
        translations_base=translations_base,
        translations_world=translations_world,
        translations_scene_local=translations_scene_local,
        translations_origin_normalized=translations_origin_normalized_2d,
        rotations_base=rotations_base,
        rotations_world=rotations_world,
        sizes=sizes,
        velocities_world=velocities_world,
        tracking_scores=np.full(frame_indices.shape[0], -1.0, dtype=np.float64),
        num_pts=np.full(frame_indices.shape[0], -1, dtype=np.int64),
        fragment_ids=fragment_ids,
        output_stem=f"{scene_name}__ego__ego",
    )


def build_track_records(
    lidar_view, *, source_dataset: str,
    ego_size_lwh: tuple = DEFAULT_EGO_SIZE_LWH,
) -> tuple[Optional[TrajectoryRecord], list[TrajectoryRecord]]:
    """Walk a sorted-by-``frame_idx`` lidar view of one scene and emit records.

    Returns ``(ego_record_or_None, [object_records, ...])``. Object
    records with fewer than 2 frames are dropped.
    """
    scene_names = lidar_view.distinct("scene_name")
    if len(scene_names) != 1:
        raise RuntimeError(
            f"build_track_records expects a single-scene view; got {scene_names!r}"
        )
    scene_name = scene_names[0]

    fields = [
        "id",
        "frame_idx",
        "m_frame_timestamp",
        "world_to_base.matrix_4x4_row_major",
        "world_to_base.translation",
        "world_to_base.quaternion_xyzw",
        "detections.detections.instance._id",
        "detections.detections.label",
        "detections.detections.location",
        "detections.detections.rotation",
        "detections.detections.dimensions",
        "detections.detections.tracking_id",
        "detections.detections.segment_index",
    ]
    (
        sample_ids, frame_idxs, m_tss,
        wtb_matrices, wtb_translations, wtb_quats,
        inst_ids_per_frame, labels_per_frame,
        locs_per_frame, rots_per_frame, dims_per_frame,
        src_tids_per_frame, src_segs_per_frame,
    ) = lidar_view.values(fields)

    # Ego trajectory
    ego_frame_idxs, ego_tokens, ego_tss, ego_trans, ego_quats = [], [], [], [], []
    for i, M in enumerate(wtb_matrices):
        if M is None or wtb_translations[i] is None or wtb_quats[i] is None:
            continue
        ego_frame_idxs.append(frame_idxs[i])
        ego_tokens.append(sample_ids[i])
        ego_tss.append(m_tss[i])
        ego_trans.append(wtb_translations[i])
        ego_quats.append(wtb_quats[i])
    ego = None
    if ego_frame_idxs:
        ego = _record_from_ego(
            scene_name=scene_name,
            source_dataset=source_dataset,
            frame_indices=ego_frame_idxs,
            sample_tokens=ego_tokens,
            timestamps_ns=ego_tss,
            wtb_translations=ego_trans,
            wtb_quats_xyzw=ego_quats,
            ego_size_lwh=tuple(ego_size_lwh),
        )

    # Per-instance buckets keyed by FO instance.id hex
    per_instance: dict[str, dict] = defaultdict(lambda: {
        "frame_indices": [], "sample_tokens": [], "timestamps_ns": [],
        "locations_base": [], "rotations_xyz_base": [], "dimensions": [],
        "wtb_matrices": [], "wtb_quats_world": [], "labels": [],
        "src_tracking_ids": [], "src_segment_indices": [],
    })

    def _pick(lst, j):
        return lst[j] if (lst is not None and j < len(lst)) else None

    for i in range(len(frame_idxs)):
        inst_ids = inst_ids_per_frame[i] or []
        labels = labels_per_frame[i] or []
        locs = locs_per_frame[i] or []
        rots = rots_per_frame[i] or []
        dims = dims_per_frame[i] or []
        src_tids = src_tids_per_frame[i] or []
        src_segs = src_segs_per_frame[i] or []
        M = wtb_matrices[i]
        q = wtb_quats[i]
        if M is None or q is None:
            continue
        for j, inst_id in enumerate(inst_ids):
            if inst_id is None:
                continue
            loc = _pick(locs, j)
            rot = _pick(rots, j)
            dim = _pick(dims, j)
            if loc is None or rot is None or dim is None:
                continue
            key = str(inst_id)
            bucket = per_instance[key]
            bucket["frame_indices"].append(frame_idxs[i])
            bucket["sample_tokens"].append(sample_ids[i])
            bucket["timestamps_ns"].append(m_tss[i])
            bucket["locations_base"].append(loc)
            bucket["rotations_xyz_base"].append(rot)
            bucket["dimensions"].append(dim)
            bucket["wtb_matrices"].append(M)
            bucket["wtb_quats_world"].append(q)
            bucket["labels"].append(_pick(labels, j) or "unknown")
            bucket["src_tracking_ids"].append(_pick(src_tids, j))
            bucket["src_segment_indices"].append(_pick(src_segs, j))

    def _first_set(seq, default):
        for v in seq:
            if v is not None:
                return v
        return default

    object_records: list[TrajectoryRecord] = []
    for inst_id, b in per_instance.items():
        if len(b["frame_indices"]) < 2:
            continue
        tracking_name = Counter(b["labels"]).most_common(1)[0][0]
        src_tid = _first_set(b["src_tracking_ids"], "")
        src_seg = _first_set(b["src_segment_indices"], 0)

        object_records.append(_record_from_object(
            scene_name=scene_name,
            source_dataset=source_dataset,
            instance_id=inst_id,
            tracking_id=str(src_tid) if src_tid is not None else "",
            segment_index=int(src_seg) if src_seg is not None else 0,
            class_run_idx=0,
            tracking_name=tracking_name,
            tracking_names_distinct=list(b["labels"]),
            frame_indices=b["frame_indices"],
            sample_tokens=b["sample_tokens"],
            timestamps_ns=b["timestamps_ns"],
            locations_base=b["locations_base"],
            rotations_xyz_base=b["rotations_xyz_base"],
            dimensions=b["dimensions"],
            wtb_matrices=b["wtb_matrices"],
            wtb_quats_world=b["wtb_quats_world"],
        ))

    return ego, object_records
