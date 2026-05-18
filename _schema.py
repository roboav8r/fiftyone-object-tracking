"""Trajectories-dataset schema + sidebar groups, plus a small helper to
apply both to a freshly created dataset.
"""

from __future__ import annotations

import fiftyone as fo
import fiftyone.core.fields as fof


# Sample-level fields on the trajectories dataset. All scalars
# (Float/Int/String/Bool + a single ListField for tracking_names_distinct);
# per-frame arrays live in the Parquet file at sample.filepath.
SAMPLE_SCHEMA: dict[str, type] = {
    # identity / back-links
    "kind": fof.StringField,
    "instance_id": fof.StringField,
    "tracking_id": fof.StringField,
    "segment_index": fof.IntField,
    "class_run_idx": fof.IntField,
    "tracking_name": fof.StringField,
    "scene_name": fof.StringField,
    "source_dataset": fof.StringField,
    "sample_token_first": fof.StringField,
    "sample_token_last": fof.StringField,

    # coverage
    "n_frames": fof.IntField,
    "duration_s": fof.FloatField,
    "frame_idx_first": fof.IntField,
    "frame_idx_last": fof.IntField,
    "is_fragmented": fof.BooleanField,
    "n_fragments": fof.IntField,
    "n_gap_frames": fof.IntField,
    "max_gap_length": fof.IntField,
    "is_stationary": fof.BooleanField,

    # position (base)
    "start_x_base": fof.FloatField,
    "start_y_base": fof.FloatField,
    "end_x_base": fof.FloatField,
    "end_y_base": fof.FloatField,
    "start_distance_m_base": fof.FloatField,
    "end_distance_m_base": fof.FloatField,
    "start_quadrant_base": fof.StringField,
    "end_quadrant_base": fof.StringField,
    "closest_approach_m_base": fof.FloatField,
    "closest_approach_frame_idx": fof.IntField,

    # position (world)
    "start_x_world": fof.FloatField,
    "start_y_world": fof.FloatField,
    "end_x_world": fof.FloatField,
    "end_y_world": fof.FloatField,
    "displacement_m": fof.FloatField,
    "path_length_m": fof.FloatField,

    # motion
    "mean_speed_m_s": fof.FloatField,
    "max_speed_m_s": fof.FloatField,
    "min_speed_m_s": fof.FloatField,
    "speed_std_m_s": fof.FloatField,

    # shape
    "heading_change_deg": fof.FloatField,
    "heading_class": fof.StringField,
    "straightness": fof.FloatField,
    "side_pass": fof.BooleanField,
    "crosses_ego_path": fof.BooleanField,

    # QC (annotation-error surfacing)
    "n_distinct_classes": fof.IntField,
    "tracking_names_distinct": fof.ListField,
    "max_step_jump_m": fof.FloatField,
    "max_gap_s": fof.FloatField,

    # AABBs flattened
    "bbox_base_x_min": fof.FloatField,
    "bbox_base_y_min": fof.FloatField,
    "bbox_base_x_max": fof.FloatField,
    "bbox_base_y_max": fof.FloatField,
    "bbox_world_x_min": fof.FloatField,
    "bbox_world_y_min": fof.FloatField,
    "bbox_world_x_max": fof.FloatField,
    "bbox_world_y_max": fof.FloatField,
}


SIDEBAR_GROUPS: list[tuple[str, list[str]]] = [
    ("tags", ["tags", "_label_tags"]),
    ("Identity", [
        "kind", "instance_id", "tracking_id",
        "segment_index", "class_run_idx",
        "tracking_name", "scene_name", "source_dataset",
        "sample_token_first", "sample_token_last",
    ]),
    ("Coverage", [
        "n_frames", "duration_s",
        "frame_idx_first", "frame_idx_last",
        "is_fragmented", "n_fragments",
        "n_gap_frames", "max_gap_length",
        "is_stationary",
    ]),
    ("Position (base)", [
        "start_x_base", "start_y_base",
        "end_x_base", "end_y_base",
        "start_distance_m_base", "end_distance_m_base",
        "start_quadrant_base", "end_quadrant_base",
        "closest_approach_m_base", "closest_approach_frame_idx",
        "bbox_base_x_min", "bbox_base_y_min",
        "bbox_base_x_max", "bbox_base_y_max",
    ]),
    ("Position (world)", [
        "start_x_world", "start_y_world",
        "end_x_world", "end_y_world",
        "displacement_m", "path_length_m",
        "bbox_world_x_min", "bbox_world_y_min",
        "bbox_world_x_max", "bbox_world_y_max",
    ]),
    ("Motion", [
        "mean_speed_m_s", "max_speed_m_s",
        "min_speed_m_s", "speed_std_m_s",
    ]),
    ("Shape", [
        "heading_change_deg", "heading_class",
        "straightness", "side_pass", "crosses_ego_path",
    ]),
    ("QC", [
        "n_distinct_classes", "tracking_names_distinct",
        "max_step_jump_m", "max_gap_s",
    ]),
    ("metadata", ["metadata"]),
]


def declare_schema(dataset: fo.Dataset) -> None:
    """Declare the trajectories sample-level schema on ``dataset`` (idempotent)."""
    for name, field_cls in SAMPLE_SCHEMA.items():
        if not dataset.has_field(name):
            dataset.add_sample_field(name, field_cls)


def set_sidebar_groups(dataset: fo.Dataset) -> None:
    """Apply the canonical sidebar grouping to ``dataset``."""
    dataset.app_config.sidebar_groups = [
        fo.SidebarGroupDocument(name=name, paths=paths)
        for name, paths in SIDEBAR_GROUPS
    ]
    dataset.save()
