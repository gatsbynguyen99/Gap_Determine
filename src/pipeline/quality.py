from typing import Dict, Tuple
import cv2
import numpy as np
from .config import QualityCfg

#Quality controlled system for raw measurement


def compute_quality(gray: np.ndarray) -> Dict[str, float]:
    #For Blur metric
    #compute Laplcian variance
    #if low variance, image is blurry, edges unreliable
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    #Saturation ratio:
    #determine the fraction of pixels near 0 or near 255
    #high saturation -> clipping: edges can shift or disappear
    sat_ratio = float(np.mean((gray <= 2) | (gray >= 253)))
    return {"blur_var": blur_var, "saturation_ratio": sat_ratio}



def confidence_gate(qcfg: QualityCfg, quality: Dict[str, float], meas_info: Dict) -> Tuple[bool, Dict]:
    passes = True
    reasons = []
    #check if blur_var >= blur_min_var
    # saturation_ratio <= max_saturation_ratio
    # valid_ratio >= min_valid_ratio
    # edge_strength >= min_edge_strength
    if quality["blur_var"] < qcfg.blur_min_var:
        passes = False
        reasons.append("too_blurry")

    if quality["saturation_ratio"] > qcfg.max_saturation_ratio:
        passes = False
        reasons.append("saturated")

    if meas_info.get("valid_ratio", 0.0) < qcfg.min_valid_ratio:
        passes = False
        reasons.append("insufficient_valid_scanlines")

    if meas_info.get("edge_strength", 0.0) < qcfg.min_edge_strength:
        passes = False
        reasons.append("weak_edges")

    return passes, {"passes": passes, "reasons": reasons}
