from typing import Dict, Optional, Tuple
import numpy as np

from .config import MeasurementCfg

#compute gap_pc robustly via sanlines

def measure_gap_from_gradmag(mag: np.ndarray, cfg: MeasurementCfg) -> Tuple[Optional[float], Dict]:
    #mag: gradient magnitude image for ROI
    #measurement_cfg: count of scanline and vertical range, minimum gap threshold(min_gap_px)
    
    """
    Scanline method:
      - For each selected y, find strongest gradient on left half and right half.
      - Gap_px = right_x - left_x.
      - Return robust median gap_px + diagnostics.
    """
    #determine the scanline band
    rh, rw = mag.shape[:2]
    y0 = int(rh * cfg.scanline_y_min)
    y1 = int(rh * cfg.scanline_y_max)
    y0 = max(0, min(y0, rh - 1))
    y1 = max(0, min(y1, rh))

    if y1 <= y0 + 1:
        return None, {"valid_ratio": 0.0, "edge_strength": 0.0, "n_valid": 0}


    #Create n_scanlines evenly spaced y_values
    ys = np.linspace(y0, y1 - 1, cfg.n_scanlines).astype(int)

    gaps = []
    strengths = []
    valid = 0

    mid = rw // 2
    #for each scaline (1 row of pixels)
    for yy in ys:
        row = mag[yy, :]
        #split the row into half left and right half
        left = row[:mid]
        right = row[mid:]

        #It enforces the physical assumption: gap is between two dominant boundaries roughly left and right of center.
        #It prevents selecting two edges on the same side
        
        #find index of maximum gradient in each half
        lx = int(np.argmax(left))
        rx = int(np.argmax(right)) + mid

        lval = float(left[lx])
        rval = float(row[rx])
        #compute the gap
        gap = rx - lx
        #validate if gap > min_gap_px
        if gap <= cfg.min_gap_px:
            continue
        
        #store gap and everage edge strength (lval + eval)/2
        gaps.append(gap)
        strengths.append((lval + rval) / 2.0)
        valid += 1

    #if no scanline return None + diagnostic info
    if valid == 0:
        return None, {"valid_ratio": 0.0, "edge_strength": 0.0, "n_valid": 0}

    gaps = np.array(gaps, dtype=np.float32)
    strengths = np.array(strengths, dtype=np.float32)

    info = {
        "valid_ratio": float(valid / len(ys)), #valid_scanlines/total_scanlines
        "edge_strength": float(np.mean(strengths)), #mean gradient strength
        "gap_px_mean": float(np.mean(gaps)),
        "gap_px_median": float(np.median(gaps)),
        "gap_px_std": float(np.std(gaps)),
        "n_valid": int(valid),
    }

    #median of gaps as final estimate
    #median can Robust to outliers (bad scanlines from shadows or clutter)
    return float(np.median(gaps)), info

def measure_gap_edges_from_mask(mask01: np.ndarray, cfg: MeasurementCfg):
    """
    Returns:
      gap_px_median, info, edges
    edges contains:
      - x_starts: list[int]
      - x_ends: list[int]
      - ys: list[int]
      - x_left_med, x_right_med
    """
    rh, rw = mask01.shape[:2]
    y0 = int(rh * cfg.scanline_y_min)
    y1 = int(rh * cfg.scanline_y_max)
    ys = np.linspace(max(0, y0), max(0, y1-1), cfg.n_scanlines).astype(int)

    widths, x_starts, x_ends, y_used = [], [], [], []
    valid = 0

    for yy in ys:
        row = mask01[yy, :]
        idx = np.where(row > 0)[0]
        if idx.size < 2:
            continue

        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        longest = max(groups, key=lambda g: g.size)

        xs, xe = int(longest[0]), int(longest[-1])
        width = xe - xs + 1
        if width <= cfg.min_gap_px:
            continue

        widths.append(width)
        x_starts.append(xs)
        x_ends.append(xe)
        y_used.append(int(yy))
        valid += 1

    if valid == 0:
        return None, {"valid_ratio": 0.0, "n_valid": 0}, {}

    widths = np.array(widths, np.float32)
    x_starts = np.array(x_starts, np.int32)
    x_ends = np.array(x_ends, np.int32)

    info = {
        "valid_ratio": float(valid / len(ys)),
        "edge_strength": 999.0,
        "gap_px_median": float(np.median(widths)),
        "gap_px_mean": float(np.mean(widths)),
        "gap_px_std": float(np.std(widths)),
        "n_valid": int(valid),
    }
    edges = {
        "x_left_med": int(np.median(x_starts)),
        "x_right_med": int(np.median(x_ends)),
        "ys": y_used,
        "x_starts": x_starts.tolist(),
        "x_ends": x_ends.tolist(),
    }
    return float(np.median(widths)), info, edges
