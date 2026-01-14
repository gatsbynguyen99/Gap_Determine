from dataclasses import dataclass
from typing import Optional
import numpy as np
from .config import ScaleCfg

#Unified interface for converting pixels to mm
#measurement module outputs gap_px
#scale module outputs mm_per_px
#gap_mm = gap_px*mm_per_px

#always returns the configured constant mm_per_px
#implement later ehen having photos calibration

class ScaleEstimator:
    def estimate_mm_per_px(self, frame_bgr: np.ndarray, roi_offset=(0, 0)) -> float:
        raise NotImplementedError


@dataclass
class FixedScaleEstimator(ScaleEstimator):
    mm_per_px: float

    def estimate_mm_per_px(self, frame_bgr: np.ndarray, roi_offset=(0, 0)) -> float:
        return float(self.mm_per_px)


class StandoffScaleEstimator(ScaleEstimator):
    """
    Placeholder: in future, use known camera-to-object distance (standoff)
    and camera intrinsics to compute mm/px for the current frame.
    """
    def __init__(self):
        pass

    def estimate_mm_per_px(self, frame_bgr: np.ndarray, roi_offset=(0, 0)) -> float:
        raise NotImplementedError("Standoff scale not implemented yet.")


class MarkerScaleEstimator(ScaleEstimator):
    """
    Placeholder: in future, detect a reference marker (e.g., known-size square)
    and compute mm/px directly from its pixel size in the frame.
    """
    def __init__(self):
        pass

    def estimate_mm_per_px(self, frame_bgr: np.ndarray, roi_offset=(0, 0)) -> float:
        raise NotImplementedError("Marker-based scale not implemented yet.")


def build_scale_estimator(cfg: ScaleCfg) -> ScaleEstimator:
    if cfg.mode == "fixed":
        return FixedScaleEstimator(mm_per_px=cfg.mm_per_px)
    if cfg.mode == "standoff":
        return StandoffScaleEstimator()
    if cfg.mode == "marker":
        return MarkerScaleEstimator()
    raise ValueError(f"Unknown scale.mode: {cfg.mode}")
