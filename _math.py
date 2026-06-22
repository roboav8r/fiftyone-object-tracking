"""Math helpers for trajectory generation.

SE(3) / quaternion primitives + per-trajectory statistics (gap stats,
step velocities, heading, origin-normalize, etc.). Kept dependency-free
beyond numpy + scipy so the helpers are easy to lift into other tooling.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


# -----------------------------------------------------------------------------
# SE(3) / quat primitives
# -----------------------------------------------------------------------------

def se3_matrix(t, q_xyzw) -> list[float]:
    """Translation + quaternion → 4×4 row-major SE(3) as a flat list[16]."""
    rot = R.from_quat(list(q_xyzw)).as_matrix()
    return [
        float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2]), float(t[0]),
        float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2]), float(t[1]),
        float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2]), float(t[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def quat_to_yaw(q_xyzw) -> float:
    """Yaw about world up (z) from a unit quaternion (xyzw)."""
    qx, qy, qz, qw = q_xyzw
    return float(math.atan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def euler_xyz_to_quat_xyzw(rx: float, ry: float, rz: float):
    """Intrinsic XYZ Euler → quaternion (qx, qy, qz, qw).

    Matches FiftyOne's convention for ``fo.Detection.rotation``
    (rotation about each axis in turn).
    """
    qx, qy, qz, qw = R.from_euler("XYZ", [rx, ry, rz]).as_quat()
    return float(qx), float(qy), float(qz), float(qw)


def apply_se3(matrix_4x4_row_major, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4×4 row-major SE(3) to an N×3 array. Returns N×3."""
    M = np.asarray(matrix_4x4_row_major, dtype=np.float64).reshape(4, 4)
    homo = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])
    return (homo @ M.T)[:, :3]


# -----------------------------------------------------------------------------
# Trajectory math
# -----------------------------------------------------------------------------

def classify_heading(deg: float) -> str:
    a = abs(deg)
    if a < 10.0:
        return "straight"
    if a < 30.0:
        return "slight_left" if deg > 0 else "slight_right"
    if a < 150.0:
        return "left" if deg > 0 else "right"
    return "u_turn"


def quadrant_base(x: float, y: float) -> str:
    """Quadrant label in base_link (x = forward, y = left). Ties → front / left."""
    fb = "front" if x >= 0 else "back"
    lr = "left" if y >= 0 else "right"
    return f"{fb}_{lr}"


def signed_crossings(values: np.ndarray) -> bool:
    """True if ``values`` changes sign (zero is treated as no crossing)."""
    if values.shape[0] < 2:
        return False
    return bool((values > 0).any() and (values < 0).any())


def gap_stats(timestamps_s: np.ndarray) -> tuple[int, int]:
    """``(n_gap_frames, max_gap_length)`` measured in median-dt cadence units.

    Robust to keyframe-stride sampling: for uniformly-spaced timestamps
    both counts are 0; for sparse keyframes within a faster underlying
    cadence the gap is reported in median-dt units regardless of the
    sweep count.
    """
    n = timestamps_s.shape[0]
    if n < 2:
        return 0, 0
    dts = np.diff(timestamps_s)
    pos = dts[dts > 0]
    if pos.size == 0:
        return 0, 0
    median_dt = float(np.median(pos))
    if median_dt <= 0:
        return 0, 0
    gap_counts = np.maximum(0, np.round(dts / median_dt).astype(np.int64) - 1)
    return int(gap_counts.sum()), int(gap_counts.max() if gap_counts.size else 0)


def origin_normalize(xy: np.ndarray) -> np.ndarray:
    """Translate to (0,0), rotate so chord (first → last) aligns with +x."""
    if xy.shape[0] < 2:
        return xy - xy[0:1] if xy.shape[0] else xy
    p = xy - xy[0:1]
    dx, dy = float(p[-1, 0]), float(p[-1, 1])
    chord = math.hypot(dx, dy)
    if chord < 1e-6:
        return p
    c, s = dx / chord, dy / chord
    rot = np.array([[c, s], [-s, c]], dtype=np.float64)
    return p @ rot.T


def heading_normalize(xy: np.ndarray, heading0: float) -> np.ndarray:
    """Translate to (0,0), rotate so initial heading ``heading0`` aligns +x.

    Unlike ``origin_normalize`` (which anchors the start→end chord), this
    anchors the trajectory's *starting orientation*, so a path that turns
    around relative to its initial heading lands behind the origin (-x)
    instead of collapsing to a forward line. ``heading0`` is a world-frame
    yaw in radians, so this pairs with world / scene_local point arrays.
    """
    if xy.shape[0] < 1:
        return xy
    p = xy - xy[0:1]
    c, s = math.cos(heading0), math.sin(heading0)
    # Rotate every point by -heading0: x' = x c + y s, y' = -x s + y c.
    rot = np.array([[c, s], [-s, c]], dtype=np.float64)
    return p @ rot.T


def resample_arclength(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a path to ``n`` points evenly spaced along its arc length.

    Only *downsamples* (returns the path unchanged when it already has
    ``<= n`` points), so it caps the per-path length that feeds the
    ``O(T^2)`` DTW without inflating short paths. Endpoints are preserved.
    Degenerate (stationary / <2-point) paths are returned unchanged.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if n <= 1 or xy.shape[0] <= n or xy.shape[0] < 2:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(d[-1])
    if total <= 0.0:
        return xy  # stationary; nothing meaningful to resample
    t = np.linspace(0.0, total, n)
    x = np.interp(t, d, xy[:, 0])
    y = np.interp(t, d, xy[:, 1])
    return np.stack([x, y], axis=1)


def heading_change_deg(xy: np.ndarray) -> float:
    """Smoothed turning angle in degrees, +CCW; wrapped to (-180, 180]."""
    n = xy.shape[0]
    if n < 4:
        return 0.0
    window = max(1, n // 4)
    start_vec = xy[window] - xy[0]
    end_vec = xy[-1] - xy[-1 - window]
    if np.linalg.norm(start_vec) < 1e-6 or np.linalg.norm(end_vec) < 1e-6:
        return 0.0
    a0 = math.atan2(float(start_vec[1]), float(start_vec[0]))
    a1 = math.atan2(float(end_vec[1]), float(end_vec[0]))
    d = math.degrees(a1 - a0)
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def step_speeds(xy: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Per-step speed magnitudes in m/s; length T (last duplicates prev)."""
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[0], dtype=np.float64)
    diffs = np.diff(xy, axis=0)
    dts = np.diff(t_s)
    safe_dts = np.where(dts > 0, dts, np.nan)
    step = np.linalg.norm(diffs, axis=1) / safe_dts
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([step, step[-1:]])


def step_velocities(xy: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Per-step (vx, vy) in m/s; length T (last row duplicates prev)."""
    if xy.shape[0] < 2:
        return np.zeros((xy.shape[0], 2), dtype=np.float64)
    diffs = np.diff(xy, axis=0)
    dts = np.diff(t_s)[:, None]
    safe_dts = np.where(dts > 0, dts, np.nan)
    vel = diffs / safe_dts
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([vel, vel[-1:]], axis=0)


def per_frame_fragment_ids(timestamps_s: np.ndarray, gap_factor: float = 2.0) -> np.ndarray:
    """Per-frame fragment index, incremented when dt > gap_factor * median_dt."""
    n = timestamps_s.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)
    dts = np.diff(timestamps_s)
    pos = dts[dts > 0]
    median_dt = float(np.median(pos)) if pos.size else 1.0
    threshold = gap_factor * median_dt
    ids = np.zeros(n, dtype=np.int32)
    cur = 0
    for i in range(1, n):
        if dts[i - 1] > threshold:
            cur += 1
        ids[i] = cur
    return ids


def timestamps_to_seconds(timestamps: Iterable[Optional[str]]) -> np.ndarray:
    """Timestamp strings → float seconds, zero-anchored at first valid value.

    Accepts pure-integer ns strings (``"1631639465431465000"``) or
    decimal-second strings (``"1631639465.431465000"``); both forms
    appear in raillabel- and dx3-derived datasets.
    """
    out: list[float] = []
    anchor: Optional[float] = None
    for s in timestamps:
        if s is None:
            out.append(float("nan"))
            continue
        if "." in s:
            v = float(s)
        else:
            v = float(int(s)) / 1e9
        if anchor is None:
            anchor = v
        out.append(v - anchor)
    arr = np.asarray(out, dtype=np.float64)
    if np.isnan(arr).any():
        idx = np.arange(arr.shape[0], dtype=np.float64)
        mask = ~np.isnan(arr)
        if mask.sum() >= 2:
            arr = np.interp(idx, idx[mask], arr[mask])
        else:
            arr = np.zeros_like(arr)
    return arr


def scalars_from_xy(xy_base: np.ndarray, xy_world: np.ndarray, step_speeds_arr: np.ndarray) -> dict:
    """Compute the trajectory's scalar metadata from per-frame 2D positions."""
    if xy_world.shape[0] >= 2:
        displacement = float(np.linalg.norm(xy_world[-1] - xy_world[0]))
        path_length = float(np.sum(np.linalg.norm(np.diff(xy_world, axis=0), axis=1)))
    else:
        displacement = 0.0
        path_length = 0.0
    mean_speed = float(np.mean(step_speeds_arr)) if step_speeds_arr.size else 0.0
    max_speed = float(np.max(step_speeds_arr)) if step_speeds_arr.size else 0.0
    min_speed = float(np.min(step_speeds_arr)) if step_speeds_arr.size else 0.0
    speed_std = float(np.std(step_speeds_arr)) if step_speeds_arr.size else 0.0
    h_change = heading_change_deg(xy_world)
    is_stationary = bool(displacement < 1.0 and max_speed < 0.5)
    straightness = float(displacement / path_length) if path_length > 1e-6 else 0.0

    base_r = np.linalg.norm(xy_base, axis=1)
    closest_idx = int(np.argmin(base_r))

    return dict(
        displacement_m=displacement, path_length_m=path_length,
        mean_speed_m_s=mean_speed, max_speed_m_s=max_speed,
        min_speed_m_s=min_speed, speed_std_m_s=speed_std,
        heading_change_deg=h_change,
        heading_class=classify_heading(h_change),
        is_stationary=is_stationary, straightness=straightness,
        closest_approach_m_base=float(base_r[closest_idx]),
        closest_approach_frame_offset=closest_idx,
        start_distance_m_base=float(base_r[0]),
        end_distance_m_base=float(base_r[-1]),
        side_pass=signed_crossings(xy_base[:, 1]),
        crosses_ego_path=signed_crossings(xy_base[:, 0]),
    )
