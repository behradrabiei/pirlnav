"""Visualize a saved world-frame semantic map.

Reads the 5-key ``.npz`` written by ``teleop_semantic_map.save_map`` (or any
other producer that follows the same schema) and renders a top-down PNG:

* world-aligned (``-Z`` = north = up, ``+X`` = east = right -- matches the
  teleop's world-frame panel)
* auto-cropped to the known-cell bounding box plus ``--margin-cells`` padding
  so the picture isn't a sea of UNKNOWN
* coloured by the same ``PALETTE`` the teleop and the policy see; UNKNOWN is
  rendered as a dark grey
* annotated with an origin marker (the world-frame anchor, i.e. the agent
  start pose at teleop time), a per-class legend listing only classes that
  actually appear in the map, and a footer with cell counts + metric size

The output PNG is written next to the input ``.npz`` by default
(``<map>_topdown.png``); override with ``--output-path``.

Usage::

    python visualize_semantic_map.py \\
        --map-path data/semantic_maps/mp3d/17DRP5sb8fy/17DRP5sb8fy.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from pirlnav.task.semantic_map import (
    FREE,
    NUM_CHANNELS,
    OBJECTNAV_CATEGORIES,
    OCCUPIED,
    PALETTE,
    UNKNOWN,
    label_map_to_rgb,
)


# Index 0 = FREE, 1 = OCCUPIED, 2..22 = the 21 goal classes (in
# OBJECTNAV_CATEGORIES order). Same convention every other consumer of
# ``global_map`` uses.
_LABEL_NAMES = ["free", "occupied"] + [name for name, _ in OBJECTNAV_CATEGORIES]
assert len(_LABEL_NAMES) == NUM_CHANNELS


def _known_bbox(
    global_map: np.ndarray, margin_cells: int
) -> Tuple[int, int, int, int]:
    """Inclusive ``(r0, r1, c0, c1)`` bbox of cells with ``label >= 0``, padded
    by ``margin_cells`` and clipped to the map bounds. Returns the full map
    if no cell is known.
    """
    valid = global_map >= 0
    if not valid.any():
        return 0, global_map.shape[0] - 1, 0, global_map.shape[1] - 1
    rows = np.where(np.any(valid, axis=1))[0]
    cols = np.where(np.any(valid, axis=0))[0]
    r0 = max(0, int(rows[0]) - margin_cells)
    r1 = min(global_map.shape[0] - 1, int(rows[-1]) + margin_cells)
    c0 = max(0, int(cols[0]) - margin_cells)
    c1 = min(global_map.shape[1] - 1, int(cols[-1]) + margin_cells)
    return r0, r1, c0, c1


def _draw_origin_marker(
    canvas: np.ndarray, origin_row_local: int, origin_col_local: int
) -> None:
    """Cross + outline circle at the (cropped) cell that corresponds to the
    map's world-frame origin (``origin_x``, ``origin_z``)."""
    h, w = canvas.shape[:2]
    if not (0 <= origin_row_local < h and 0 <= origin_col_local < w):
        return
    cv2.circle(canvas, (origin_col_local, origin_row_local), 8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (origin_col_local, origin_row_local), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2,
                   line_type=cv2.LINE_AA)


def _draw_legend(
    canvas: np.ndarray, present_labels: np.ndarray
) -> None:
    """Bottom-left legend with one swatch + name per label actually present in
    the cropped map. Always lists UNKNOWN first if any cell is UNKNOWN.
    """
    h, w = canvas.shape[:2]
    entries = []
    if (present_labels == UNKNOWN).any():
        entries.append(("unknown", (20, 20, 20)))
    for v in sorted(int(x) for x in np.unique(present_labels) if x >= 0):
        entries.append((_LABEL_NAMES[v], tuple(int(c) for c in PALETTE[v])))
    if not entries:
        return

    pad, sw, line_h = 8, 14, 18
    legend_h = pad * 2 + line_h * len(entries)
    legend_w = pad * 2 + sw + 6 + 130
    x0, y0 = pad, h - pad - legend_h
    if y0 < 0:
        return
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    for i, (name, color) in enumerate(entries):
        cy = y0 + pad + i * line_h
        cv2.rectangle(canvas, (x0 + pad, cy), (x0 + pad + sw, cy + line_h - 4),
                      color, -1)
        cv2.putText(canvas, name, (x0 + pad + sw + 6, cy + line_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)


def render_topdown(
    global_map: np.ndarray,
    origin_x: float,
    origin_z: float,
    resolution: float,
    margin_cells: int = 12,
) -> np.ndarray:
    """Crop, recolour, and annotate the world-frame label map. Returns RGB."""
    r0, r1, c0, c1 = _known_bbox(global_map, margin_cells)
    crop = global_map[r0 : r1 + 1, c0 : c1 + 1]
    canvas = label_map_to_rgb(crop)  # already RGB

    # The full global map is anchored so that cell (H_g/2, W_g/2) corresponds
    # to (origin_x, origin_z) in world space. Translate that to the crop.
    h_g, w_g = global_map.shape
    origin_row_full = h_g // 2
    origin_col_full = w_g // 2
    _draw_origin_marker(canvas, origin_row_full - r0, origin_col_full - c0)

    _draw_legend(canvas, crop)

    n_known = int((crop >= 0).sum())
    n_free = int((crop == FREE).sum())
    n_occ = int((crop == OCCUPIED).sum())
    n_goal = int((crop >= 2).sum())
    h_m = crop.shape[0] * resolution
    w_m = crop.shape[1] * resolution
    cv2.putText(
        canvas,
        f"crop={crop.shape[1]}x{crop.shape[0]} cells  "
        f"({w_m:.1f}m x {h_m:.1f}m, res={resolution * 100:.1f}cm)  "
        f"known={n_known} free={n_free} occ={n_occ} goal={n_goal}   "
        "(up = -Z / north)",
        (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-path", type=Path,
        default=Path("data/semantic_maps/mp3d/17DRP5sb8fy/17DRP5sb8fy.npz"),
        help="Path to the npz produced by teleop_semantic_map.save_map.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Where to write the PNG. Defaults to '<map>_topdown.png' next "
             "to --map-path.",
    )
    parser.add_argument(
        "--margin-cells", type=int, default=12,
        help="Extra padding around the known-cell bbox before cropping.",
    )
    args = parser.parse_args()

    if not args.map_path.is_file():
        raise FileNotFoundError(f"No such map: {args.map_path}")

    data = np.load(str(args.map_path), allow_pickle=True)
    global_map = np.asarray(data["global_map"])
    origin_x = float(data["origin_x"])
    origin_z = float(data["origin_z"])
    resolution = float(data["resolution"])
    scene_id = str(data["scene_id"])
    print(
        f"Loaded {global_map.shape} {global_map.dtype} map  "
        f"resolution={resolution} m/cell  "
        f"origin=({origin_x:.3f}, {origin_z:.3f})  "
        f"scene={scene_id}"
    )
    print(
        f"  known cells: {int((global_map >= 0).sum())}/{global_map.size}  "
        f"goal-class cells: {int((global_map >= 2).sum())}"
    )

    rgb = render_topdown(
        global_map, origin_x, origin_z, resolution,
        margin_cells=args.margin_cells,
    )
    out = args.output_path or args.map_path.with_name(args.map_path.stem + "_topdown.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
