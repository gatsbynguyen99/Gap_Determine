import cv2
import numpy as np
from typing import Optional, Dict

def _percentile_valid(depth_mm: np.ndarray, q: float) -> Optional[float]:
    d = depth_mm[(depth_mm > 0)]
    if d.size < 200:
        return None
    return float(np.percentile(d.astype(np.float32), q))

def find_two_pipes_and_gap_from_depth(
    depth_mm: np.ndarray,
    near_percentile: float = 20.0,
    band_mm: float = 60.0,
    min_area: int = 500,
) -> Optional[Dict]:
    """
    Detect two nearest large components (pipes) using depth and compute gap edges.

    Returns dict in ROI coords:
      pipe1: (x0, x1, y0, y1)   # left pipe bbox
      pipe2: (x0, x1, y0, y1)   # right pipe bbox
      gap_x_left, gap_x_right, gap_px_depth
      near_depth_mm
    """
    H, W = depth_mm.shape[:2]
    near_d = _percentile_valid(depth_mm, near_percentile)
    if near_d is None:
        return None

    d = depth_mm.astype(np.float32)
    valid = d > 0

    # "Pipe mask": pixels near the near surface depth
    pipe_mask = valid & (np.abs(d - near_d) <= band_mm)
    pipe_u8 = (pipe_mask.astype(np.uint8) * 255)

    # Morph cleanup
    k = np.ones((7, 7), np.uint8)
    pipe_u8 = cv2.morphologyEx(pipe_u8, cv2.MORPH_CLOSE, k, iterations=2)
    pipe_u8 = cv2.morphologyEx(pipe_u8, cv2.MORPH_OPEN,  k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(pipe_u8, connectivity=8)
    if num <= 2:
        return None

    comps = []
    for cid in range(1, num):
        x, y, w, h, area = stats[cid]
        if area >= min_area:
            comps.append((area, cid, x, y, w, h))

    if len(comps) < 2:
        return None

    comps.sort(reverse=True, key=lambda t: t[0])
    top2 = comps[:2]

    boxes = []
    for _, cid, x, y, w, h in top2:
        boxes.append((x, x + w - 1, y, y + h - 1, cid))

    # left-to-right order
    boxes.sort(key=lambda b: b[0])
    (x0a, x1a, y0a, y1a, _) = boxes[0]
    (x0b, x1b, y0b, y1b, _) = boxes[1]

    gap_left = int(x1a)
    gap_right = int(x0b)
    gap_px = int(gap_right - gap_left)

    if gap_px <= 0:
        return None

    return {
        "pipe1": (int(x0a), int(x1a), int(y0a), int(y1a)),
        "pipe2": (int(x0b), int(x1b), int(y0b), int(y1b)),
        "gap_x_left": gap_left,
        "gap_x_right": gap_right,
        "gap_px_depth": gap_px,
        "near_depth_mm": float(near_d),
    }

def build_gap_mask_from_edges(
    H: int,
    W: int,
    x_left: int,
    x_right: int,
    y_min_frac: float = 0.30,
    y_max_frac: float = 0.90
) -> np.ndarray:
    """
    Build a stable gap mask (0/1) by filling between x_left..x_right over a vertical band.

    Inputs are ROI coordinates.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    y0 = int(H * y_min_frac)
    y1 = int(H * y_max_frac)

    x_left = int(max(0, min(x_left, W - 1)))
    x_right = int(max(0, min(x_right, W)))

    if y1 <= y0:
        return mask

    if x_right > x_left:
        mask[y0:y1, x_left:x_right] = 1

    return mask
