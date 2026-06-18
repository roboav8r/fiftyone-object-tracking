"""Hierarchical clustering of trajectories from a DTW distance matrix.

Turns the pairwise DTW matrix (``_dtw.py``) into a binary merge tree
(scipy ``linkage``), cuts it into flat clusters (``fcluster``), and
extracts the dendrogram line geometry (``dendrogram(no_plot=True)``) so
the panel can draw an interactive dendrogram without a charting library
or a server-side matplotlib render.

Kept dependency-light (numpy + scipy); reuses ``origin_normalize`` from
``_math`` for ego-relative shape normalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from ._math import origin_normalize, resample_arclength

# Which per-frame point array a tracklet exposes for each reference frame.
_FRAME_KEYS = {
    "base": "xy_base",
    "world": "xy_world",
    "scene_local": "xy_scene_local",
}


def tracks_to_arrays(
    scene_tracklets,
    *,
    frame_key: str = "world",
    normalize: bool = True,
    resample: int = 0,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    """Build clusterable ``(T, 2)`` arrays from ``(scene_name, tracklet)`` pairs.

    Returns ``(members, arrays, skipped)`` where ``arrays[i]`` is the path
    for ``members[i]`` and each member is ``{"scene_name", "track_idx"}``.
    ``skipped`` lists the same dicts for tracklets dropped because they are
    single-point (``T < 2``) or non-finite — ``squareform(checks=True)``
    would reject NaNs. Members are ``(scene, track_idx)`` pairs so a single
    clustering can pool trajectories from several scenes (e.g. every run's
    ego path), and the row→member map resolves back to the right tracklet
    for select / tag / export.

    Class / ego membership is the CALLER's decision (filter before calling);
    this builder no longer special-cases ego. ``resample > 0`` arc-length-
    downsamples each path to at most that many points (DTW is
    ``O(T_a * T_b)`` per pair; sampling-invariance means clusters barely
    change).
    """
    xy_field = _FRAME_KEYS.get(frame_key, "xy_world")
    members: list[dict] = []
    arrays: list[np.ndarray] = []
    skipped: list[dict] = []
    for scene, t in scene_tracklets:
        key = {"scene_name": scene, "track_idx": int(t.get("track_idx", -1))}
        xy = np.asarray(t.get(xy_field) or [], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] < 2 or not np.all(np.isfinite(xy)):
            skipped.append(key)
            continue
        if resample:
            xy = resample_arclength(xy, resample)
        if normalize:
            xy = origin_normalize(xy)
        members.append(key)
        arrays.append(xy)
    return members, arrays, skipped


def _maxclust_cut_height(heights: np.ndarray, k: int, nobs: int) -> float:
    """Dendrogram height that cuts the tree into ``k`` clusters.

    ``heights`` is the ascending list of the ``nobs - 1`` merge heights.
    For ``2 <= k <= nobs - 1`` the cut sits midway between the last merge
    kept and the first merge dropped; the degenerate ends fall just
    below the smallest / just above the largest merge.
    """
    if heights.size == 0:
        return 0.0
    if k <= 1:
        return float(heights[-1] * 1.05)
    if k >= nobs:
        return float(heights[0] * 0.5)
    lo = heights[nobs - k - 1]
    hi = heights[nobs - k]
    return float((lo + hi) / 2.0)


def cluster_from_matrix(
    dist_matrix: np.ndarray,
    *,
    method: str = "average",
    num_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> dict:
    """Cluster a pairwise distance matrix; return a JSON-able result.

    Cuts by target count (``num_clusters`` → ``criterion="maxclust"``,
    clamped to ``[1, nobs]``) or by height (``distance_threshold`` →
    ``criterion="distance"``). The returned ``threshold`` is the height
    of the cut, which the panel renders as the draggable cut line.

    Returns ``{Z, labels, n_clusters, method, criterion, threshold,
    icoord, dcoord, leaves, color_list}``. ``labels`` is 1-based and
    indexed by matrix-row order (NOT dendrogram leaf order).
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    nobs = dist.shape[0]
    if nobs < 2:
        raise ValueError("need >= 2 observations to cluster")

    # Condensed upper triangle; checks=True validates symmetry, zero
    # diagonal, and finiteness (callers scrub non-finite paths upstream).
    condensed = squareform(dist, checks=True)
    Z = linkage(condensed, method=method)
    heights = np.sort(Z[:, 2])

    if distance_threshold is not None:
        criterion = "distance"
        threshold = float(distance_threshold)
        labels = fcluster(Z, t=threshold, criterion="distance")
    else:
        criterion = "maxclust"
        k = max(1, min(int(num_clusters if num_clusters is not None else 4), nobs))
        labels = fcluster(Z, t=k, criterion="maxclust")
        threshold = _maxclust_cut_height(heights, k, nobs)

    dn = dendrogram(Z, no_plot=True, color_threshold=threshold)

    return {
        "Z": [[float(v) for v in row] for row in Z],
        "labels": [int(x) for x in labels.tolist()],
        "n_clusters": int(len(set(int(x) for x in labels.tolist()))),
        "method": method,
        "criterion": criterion,
        "threshold": float(threshold),
        "icoord": [[float(v) for v in seg] for seg in dn["icoord"]],
        "dcoord": [[float(v) for v in seg] for seg in dn["dcoord"]],
        "leaves": [int(x) for x in dn["leaves"]],
        "color_list": list(dn["color_list"]),
    }
