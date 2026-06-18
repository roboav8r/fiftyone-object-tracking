"""Dynamic Time Warping distance for 2D trajectory point sequences.

DTW is a distance between two variable-length ordered point sequences
that is robust to differences in speed/sampling along the path (one
object drove the same route slower than another). It feeds the
pairwise distance matrix that ``_clustering.py`` turns into a
hierarchical-clustering dendrogram.

Lifted from the customer reference
``dtw_hierarchical_clustering_demo.py`` and kept dependency-light
(numpy + scipy only) so it is easy to reuse — e.g. a future
``find_similar_trajectories`` DTW nearest-neighbor operator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(a: np.ndarray, b: np.ndarray, *, band: Optional[int] = None) -> float:
    """Exact DTW distance between two ``(N, 2)`` and ``(M, 2)`` sequences.

    Builds the cumulative-cost recurrence
    ``D[i, j] = ||a[i-1] - b[j-1]|| + min(D[i-1, j], D[i, j-1], D[i-1, j-1])``
    and returns ``D[N, M]`` — the cheapest way to warp the entire first
    sequence onto the entire second, preserving point order. This is
    what makes DTW robust to different sampling rates/speeds.

    The inner column update is vectorized: the two previous-row terms
    (``D[i-1, j]`` and ``D[i-1, j-1]``) are minimized across the whole
    row in one numpy op; only the left carry (``D[i, j-1]``) stays a
    scalar scan, since it is the one genuinely sequential dependency.

    ``band`` is an optional Sakoe-Chiba radius: when set, only cells
    within ``±band`` columns of the diagonal are filled (the rest stay
    ``inf``), cutting the per-row work from ``O(M)`` to ``O(band)``. The
    radius is widened to ``max(band, |N - M|)`` so the warping path
    endpoints stay reachable.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("DTW inputs must be 2D (T, D) arrays")
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    # C[i, j] = ||a[i] - b[j]||; cost of row i is C[i - 1] (0-indexed).
    C = cdist(a, b)

    radius = None if band is None else max(int(band), abs(n - m))

    # prev/curr are length m+1; column 0 is the virtual j=0 boundary.
    # Row 0: D[0, 0] = 0, D[0, j>0] = inf.
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        cost_row = C[i - 1]                       # length m, indexed j-1
        updiag = np.minimum(prev[1:], prev[:-1])  # updiag[k] = min(prev[k+1], prev[k])

        if radius is None:
            j_lo, j_hi = 1, m
        else:
            center = i * m / n
            j_lo = max(1, int(np.floor(center - radius)))
            j_hi = min(m, int(np.ceil(center + radius)))

        left = np.inf
        for j in range(j_lo, j_hi + 1):
            best = updiag[j - 1]                  # min(D[i-1, j], D[i-1, j-1])
            if left < best:                       # D[i, j-1]
                best = left
            val = cost_row[j - 1] + best
            curr[j] = val
            left = val
        prev = curr

    return float(prev[m])


def pairwise_dtw_matrix(
    trajs: list[np.ndarray],
    *,
    band: Optional[int] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Symmetric ``(N, N)`` DTW distance matrix for an ordered list of paths.

    Row/column ``i`` corresponds to ``trajs[i]`` — the caller's ordering
    is preserved (no sorting), so the result maps straight back to
    whatever index the caller tracks. Diagonal is zero.

    This is the ``O(N^2)`` all-pairs step (each pair is itself
    ``O(len(a) * len(b))``). ``n_jobs != 1`` *attempts* joblib over the
    upper triangle, but falls back to a serial loop on ANY failure —
    joblib missing, or its subprocess pool failing to start/serialize
    (common when this code runs inside a server operator worker or a
    dynamically-imported plugin module, where loky can't re-import the
    task function). Correctness never depends on the parallel path.
    """
    n = len(trajs)
    dist = np.zeros((n, n), dtype=np.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return dist

    arrs = [np.asarray(t, dtype=np.float64) for t in trajs]

    results = None
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(dtw_distance)(arrs[i], arrs[j], band=band) for i, j in pairs
            )
        except Exception:
            results = None  # any parallel failure → fall back to serial
    if results is None:
        results = [dtw_distance(arrs[i], arrs[j], band=band) for i, j in pairs]

    for (i, j), d in zip(pairs, results):
        dist[i, j] = d
        dist[j, i] = d
    return dist
