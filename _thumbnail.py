"""Server-side BEV thumbnail rendering for trajectory FO samples.

The trajectories dataset's grid uses FO's built-in image renderer.
That means each FO sample's ``filepath`` is a PNG that's rendered
once at build time by :func:`render_trajectory_thumbnail` here.

No client-side JS renderer; no per-cell server round-trip; no
PyArrow / Parquet dependencies.

Plot convention (matches the prior interactive SampleRenderer)::

    image-up    = +x_base   (robot forward)
    image-left  = +y_base   (robot left)
    equal aspect, dark background

For object trajectories the ego rectangle is drawn at origin (using
``ego_size_lwh_m``). For ego trajectories the path itself is the
ego — no rectangle — and ``x_scene_local`` / ``y_scene_local`` are
plotted so the path isn't a single dot at the base-frame origin.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import matplotlib

# Headless backend — must be set before any pyplot import.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ._palette import color_for
from ._records import TrajectoryRecord


_SIZE_PX = 512
_DPI = 100
_BG_HEX = "#0a0a0a"
_AXIS_HEX = "#2a2a2a"
_EGO_FILL_RGBA = (0.0, 1.0, 1.0, 0.18)
_EGO_EDGE_HEX = "#00ffff"
_ORIGIN_DOT_HEX = "#2bff7f"


def _trajectory_xy(record: TrajectoryRecord) -> tuple[np.ndarray, np.ndarray]:
    """Pick the right per-frame XY arrays for this trajectory kind."""
    if record.kind == "ego":
        # Ego is the origin of its own base frame; plot scene-local
        # translations instead so the path isn't a degenerate dot.
        return (
            record.translations_scene_local[:, 0],
            record.translations_scene_local[:, 1],
        )
    return record.translations_base[:, 0], record.translations_base[:, 1]


def render_trajectory_thumbnail(
    record: TrajectoryRecord,
    outpath: str,
    *,
    ego_size_lwh: tuple[float, float, float] = (24.0, 2.9, 4.0),
    size_px: int = _SIZE_PX,
    dpi: int = _DPI,
) -> None:
    """Render one trajectory to a PNG at ``outpath``.

    The plot convention matches the dataloader-side orthographic
    thumbnails (image-up = +x_base, image-left = +y_base), so the
    trajectories grid and the source-dataset grid read consistently.
    """
    x_base, y_base = _trajectory_xy(record)
    color = color_for(record.tracking_name)

    # Bounds from finite trajectory points + 1 m pad, min half-extent 5 m.
    finite = np.isfinite(x_base) & np.isfinite(y_base)
    if finite.any():
        xs = x_base[finite]
        ys = y_base[finite]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
    else:
        x_min = x_max = y_min = y_max = 0.0

    # Object cells force-include the ego footprint at origin so the robot
    # is always visible relative to the path.
    if record.kind != "ego":
        half_L = ego_size_lwh[0] / 2
        half_W = ego_size_lwh[1] / 2
        x_min = min(x_min, -half_L); x_max = max(x_max, half_L)
        y_min = min(y_min, -half_W); y_max = max(y_max, half_W)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_x = max(5.0, (x_max - x_min) / 2 + 1.0)
    half_y = max(5.0, (y_max - y_min) / 2 + 1.0)

    fig = plt.figure(
        figsize=(size_px / dpi, size_px / dpi), dpi=dpi, facecolor=_BG_HEX,
    )
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed; no margins
    ax.set_facecolor(_BG_HEX)
    ax.set_aspect("equal", adjustable="box")
    # image-up = +x_base → matplotlib y-axis +.
    # image-left = +y_base → matplotlib x-axis inverted (so +y_base lands left).
    ax.set_xlim(cy + half_y, cy - half_y)  # inverted: high y_base on left
    ax.set_ylim(cx - half_x, cx + half_x)  # +x_base up

    # Faint axes through origin.
    ax.axhline(0.0, color=_AXIS_HEX, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=_AXIS_HEX, linewidth=0.8, zorder=1)

    # Ego rectangle at origin (object kind only).
    if record.kind != "ego":
        half_L = ego_size_lwh[0] / 2
        half_W = ego_size_lwh[1] / 2
        # Rectangle plotted in (y_base, x_base): width = full W (along y),
        # height = full L (along x). Anchor at (-half_W, -half_L).
        ax.add_patch(Rectangle(
            (-half_W, -half_L), 2 * half_W, 2 * half_L,
            facecolor=_EGO_FILL_RGBA, edgecolor=_EGO_EDGE_HEX,
            linewidth=1.0, zorder=2,
        ))

    # Per-fragment polylines + dashed bridges across gaps.
    fragment_ids = record.fragment_ids if record.fragment_ids is not None \
        else np.zeros(x_base.shape[0], dtype=np.int32)
    unique_fids = sorted({int(f) for f in fragment_ids})
    last_end_xy: Optional[tuple[float, float]] = None
    for fid in unique_fids:
        idxs = np.where(fragment_ids == fid)[0]
        if len(idxs) == 0:
            continue
        xs = x_base[idxs]
        ys = y_base[idxs]
        # Map to plot coords (x_plot, y_plot) = (y_base, x_base).
        if last_end_xy is not None:
            # Dashed bridge from the previous fragment's last point to
            # this fragment's first point.
            ax.plot(
                [last_end_xy[0], ys[0]], [last_end_xy[1], xs[0]],
                color=color, linewidth=1.0, alpha=0.6,
                linestyle="--", zorder=3,
            )
        if len(idxs) == 1:
            ax.plot(ys[0], xs[0], marker="o", color=color, markersize=2,
                    zorder=4)
        else:
            ax.plot(ys, xs, color=color, linewidth=1.6, zorder=4)
        last_end_xy = (float(ys[-1]), float(xs[-1]))

    # Start o, end x — only meaningful when the trajectory has ≥ 1 frame.
    finite_idxs = np.where(finite)[0]
    if len(finite_idxs) >= 1:
        i0 = int(finite_idxs[0])
        ax.plot(
            y_base[i0], x_base[i0], marker="o", markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=1.6, markersize=8,
            linestyle="None", zorder=5,
        )
    if len(finite_idxs) >= 2:
        iN = int(finite_idxs[-1])
        if iN != i0:
            ax.plot(
                y_base[iN], x_base[iN], marker="x",
                markeredgecolor=color, markeredgewidth=1.6, markersize=7,
                linestyle="None", zorder=5,
            )

    # Origin dot (ego center).
    ax.plot(0.0, 0.0, marker="o", color=_ORIGIN_DOT_HEX,
            markeredgecolor=_BG_HEX, markeredgewidth=0.7,
            markersize=5, linestyle="None", zorder=6)

    # Strip every tick / spine.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=dpi, facecolor=_BG_HEX,
                edgecolor="none", pad_inches=0)
    plt.close(fig)
