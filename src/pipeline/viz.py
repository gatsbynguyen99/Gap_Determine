import os
from typing import Optional, Tuple
import cv2
import numpy as np


def _draw_vline(img, x: int, color, thickness: int = 2, dashed: bool = False):
    h = img.shape[0]
    x = int(x)
    if not dashed:
        cv2.line(img, (x, 0), (x, h - 1), color, thickness)
        return
    step = 14
    dash = 7
    for y in range(0, h, step):
        y2 = min(y + dash, h - 1)
        cv2.line(img, (x, y), (x, y2), color, thickness)


def _short_id(frame_id: str) -> str:
    """
    Example:
      '17cm_20260113_231738_no_direct_light' -> '17cm_231738'
      'captures_20260113_024736' -> 'cap_024736'
    """
    s = os.path.basename(str(frame_id))
    s = s.replace(" ", "")

    parts = s.split("_")
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        # distance + HHMMSS
        prefix = parts[0]
        hhmmss = parts[2][-6:]
        return f"{prefix}_{hhmmss}"

    if s.startswith("captures_") and len(parts) >= 2:
        return f"cap_{parts[1][-6:]}"

    # fallback: last 12 chars
    return s[-12:]


def save_overlay(
    out_dir: str,
    frame_id: str,
    frame_bgr: np.ndarray,
    roi_offset: Tuple[int, int],
    roi_size: Optional[Tuple[int, int]],
    ok: bool,
    overlay_index: int,

    # ML edges (solid)
    ml_gap_x_left_full: Optional[int] = None,
    ml_gap_x_right_full: Optional[int] = None,
    ml_gap_px: Optional[float] = None,

    # Depth edges (dashed)
    depth_gap_x_left_full: Optional[int] = None,
    depth_gap_x_right_full: Optional[int] = None,
    depth_gap_px: Optional[float] = None,

    # Optional: pipes x bands
    pipe1_x0_full: Optional[int] = None,
    pipe1_x1_full: Optional[int] = None,
    pipe2_x0_full: Optional[int] = None,
    pipe2_x1_full: Optional[int] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vis = frame_bgr.copy()

    sid = _short_id(frame_id)

    # ROI rectangle (yellow)
    if roi_size is not None:
        x, y = roi_offset
        w, h = roi_size
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # pipe bands (optional)
        y0, y1 = y, y + h
        def band(x0, x1, label: str):
            if x0 is None or x1 is None:
                return

            x0 = int(x0); x1 = int(x1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)

            # Put label near the top of the band
            text = f"{label}: [{x0},{x1}]"
            # Place slightly inside band; clamp to image
            tx = max(5, min(x0 + 5, vis.shape[1] - 200))
            ty = max(25, y0 + 25)
            cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        band(pipe1_x0_full, pipe1_x1_full, "Pipe1")
        band(pipe2_x0_full, pipe2_x1_full, "Pipe2")


    # Depth edges (dashed): Red=left, Green=right
    if depth_gap_x_left_full is not None:
        _draw_vline(vis, depth_gap_x_left_full, (0, 0, 255), 2, dashed=True)
    if depth_gap_x_right_full is not None:
        _draw_vline(vis, depth_gap_x_right_full, (0, 255, 0), 2, dashed=True)

    # ML edges (solid): Cyan=left, Magenta=right
    if ml_gap_x_left_full is not None:
        _draw_vline(vis, ml_gap_x_left_full, (255, 255, 0), 2, dashed=False)
    if ml_gap_x_right_full is not None:
        _draw_vline(vis, ml_gap_x_right_full, (255, 0, 255), 2, dashed=False)

    # Compute error if both px exist
    err = None
    if ml_gap_px is not None and depth_gap_px is not None:
        err = abs(float(ml_gap_px) - float(depth_gap_px))

    # Text (two lines only)
    line1 = f"id={sid} ok={ok} depth_px={depth_gap_px if depth_gap_px is not None else 'NA'}"
    line2 = f"ML_px={ml_gap_px if ml_gap_px is not None else 'NA'} err={err if err is not None else 'NA'}"

    cv2.putText(vis, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    cv2.putText(vis, line2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

    # Optional tiny legend (single line) in bottom-left
    legend = "ML: cyan/magenta | Depth: red/green dashed | ROI: yellow"
    cv2.putText(vis, legend, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    path = os.path.join(out_dir, f"overlay_{overlay_index:04d}.jpg")
    cv2.imwrite(path, vis)
    return path
