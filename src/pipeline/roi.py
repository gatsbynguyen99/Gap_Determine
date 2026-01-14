from typing import Tuple
import numpy as np
from .config import RoiCfg

#Crop the image to the region where the gap exists to:
# reduce false edges
# increase speed
# mprove measurement stability
def apply_roi(frame: np.ndarray, roi: RoiCfg) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Input: full BGR image, ROICfg with coordinates
    Returns (cropped_frame, (offset_x, offset_y)).
    If roi is disabled or invalid, returns the original frame and offset (0,0).
    """
    h, w = frame.shape[:2]
    if not roi.enabled:
        return frame, (0, 0)

    x, y, rw, rh = roi.x, roi.y, roi.w, roi.h

    # If w/h are 0, treat as full frame (still "enabled", but no crop)
    if rw <= 0 or rh <= 0:
        return frame, (0, 0)

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    #output cropped image if ROI enabled otherwise original
    #(offset_x, offset_y): where the crop starts in the original frame

    cropped = frame[y:y+rh, x:x+rw]

    return cropped, (x, y)
