#Baseline: undistort -> CLAHE -> edges -> scanline gap -> mm conversion -> confidence
# Back up basline: segmentation -> boundary distance -> mm conversion -> covnersion


import os
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np

#config
@dataclass
class GapConfig:
    roi: Optional[Tuple[int, int, int, int]] = None # (x, y, w, h): None = full frame
    n_scanlines: int = 80 
    blur_min_var: float = 60.0 #below this blur_min_var will be too blurry
    min_valid_ratio: float = 0.70 #fraction of scanlines with 2 edges found
    min_edge_strength: float = 10.0 #average gradient madnitude threshold
    max_saturation_ration: float = 0.03
    mm_per_px: float = 0.05 #replace later by calibration/marker/standoff later

    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    canny_low: int = 40
    canny_high: int = 120


#synthetic generator for pipeline dev

def make_synthetic_gap(
        width=640, height=480, gap_px=40,
        tilt_px=0, brightness=180, shadow_strength=0.25,
        noise_sigma=8, blur_ksize=0
) -> np.ndarray:
    
    """
    Creates a synthetic image with two vertical "pipe edges" separated by gap_px.
    Simulates lighting gradients and noise so we can test robustness.
    
    """
    img = np.full((height,width), brightness, dtype=np.uint8)

    #add horizontal illumination gradient(simulated lighting direction)
    x = np.linspace(0,1,width, dtype=np.float32)
    grad = (1.0 - shadow_strength) + shadow_strength * x #brighter to the right
    img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    #define edges (slightly titled)
    cx = width // 2
    left_x0 = cx - gap_px // 2
    right_x0 = cx + gap_px // 2

    #drraw edges as darker lines; tilt makes the edges shift with y
    for y in range(height):
        dx = int(y / max(1, height - 1)* tilt_px)
        lx = np.clip(left_x0 + dx, 0, width -1)
        rx = np.clip(right_x0 + dx, 0, width -1)
        img[y, lx:lx+2] = 40
        img[y, rx:x:=rx+2] = 40
    


    #add noise:
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    #blur - optional
    if blur_ksize and blur_ksize % 2 == 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


#preproccessing

def preprocess(frame_bgr: np.ndarray, cfg: GapConfig) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
    gray = clahe.apply(gray)

    # Light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

#Quality metric
#threshold to check the acceptable quality of the fame
def compute_quality(gray: np.ndarray) -> Dict[str, float]:
    # Blur metric: variance of Laplacian (higher = sharper)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    # Saturation ratio (if many pixels are near 0 or 255, edges can be unreliable)
    sat = np.mean((gray <= 2) | (gray >= 253))

    return {"blur_var": blur_var, "saturation_ratio": float(sat)}


# Gap measurement via scanlines
def find_gap_px(gray: np.ndarray, cfg: GapConfig) -> Tuple[Optional[float], Dict]:
    h, w = gray.shape

    # Optional ROI
    if cfg.roi is not None:
        x, y, rw, rh = cfg.roi
        x = max(0, x); y = max(0, y)
        rw = min(rw, w - x); rh = min(rh, h - y)
        gray_roi = gray[y:y+rh, x:x+rw]
    else:
        x, y = 0, 0
        gray_roi = gray

    rh, rw = gray_roi.shape

    # Gradient magnitude (more stable than raw intensity under lighting changes)
    gx = cv2.Scharr(gray_roi, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_roi, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # Choose evenly spaced scanlines
    ys = np.linspace(int(rh * 0.15), int(rh * 0.85), cfg.n_scanlines).astype(int)

    gaps = []
    strengths = []
    valid = 0

    for yy in ys:
        row = mag[yy, :]

        # Split left/right halves; pick strongest edge on each half
        mid = rw // 2
        left = row[:mid]
        right = row[mid:]

        lx = int(np.argmax(left))
        rx = int(np.argmax(right)) + mid

        lval = float(left[lx])
        rval = float(row[rx])

        # Require edges to be reasonably strong
        if lval < 1e-6 or rval < 1e-6:
            continue

        # Basic sanity: edges should be separated by some minimum
        gap = rx - lx
        if gap <= 2:
            continue

        gaps.append(gap)
        strengths.append((lval + rval) / 2.0)
        valid += 1

    if valid == 0:
        return None, {"valid_ratio": 0.0, "edge_strength": 0.0, "n_valid": 0}

    gaps = np.array(gaps, dtype=np.float32)
    strengths = np.array(strengths, dtype=np.float32)

    info = {
        "valid_ratio": float(valid / len(ys)),
        "edge_strength": float(np.mean(strengths)),
        "gap_px_mean": float(np.mean(gaps)),
        "gap_px_median": float(np.median(gaps)),
        "gap_px_std": float(np.std(gaps)),
        "n_valid": int(valid),
    }

    # Use median as robust estimate
    return float(np.median(gaps)), info

def compute_confidence(cfg: GapConfig, q: Dict, info: Dict) -> Tuple[bool, Dict]:
    passes = True
    reasons = []

    if q["blur_var"] < cfg.blur_min_var:
        passes = False
        reasons.append("too_blurry")

    if q["saturation_ratio"] > cfg.max_saturation_ratio:
        passes = False
        reasons.append("saturated")

    if info.get("valid_ratio", 0.0) < cfg.min_valid_ratio:
        passes = False
        reasons.append("insufficient_valid_scanlines")

    if info.get("edge_strength", 0.0) < cfg.min_edge_strength:
        passes = False
        reasons.append("weak_edges")

    return passes, {"passes": passes, "reasons": reasons}

#Runner 

def run_on_frames(frames: List[np.ndarray], cfg: GapConfig) -> List[Dict] :
    results = []
    for i, frame in enumerate(frames):
        gray = preprocess(frame, cfg)
        q = compute_quality(gray)

        gap_px, info = find_gap_px(gray, cfg)
        if gap_px is None:
            results.append({"id": i, "ok": False, "reason": "no_gap_found", **q})
            continue
        
        ok, conf = compute_confidence(cfg, q, info)

        gap_mm = gap_px * cfg.mm_per_px
        results.append({
            "id": i,
            "ok": ok,
            "gap_px": gap_px,
            "gap_mm": gap_mm,
            **q,
            **info,
            **conf
        })

        return results
    



if __name__ == "__main__":
    cfg = GapConfig(
        roi=None,           # set later when know where the gap appears
        mm_per_px=0.05,     # placeholder; replace later
    )

    # Synthetic test batch (no photos needed yet)
    synth = []
    for gap in [20, 30, 40, 50]:
        for brightness in [120, 180, 220]:
            for shadow in [0.0, 0.25, 0.5]:
                synth.append(make_synthetic_gap(
                    gap_px=gap,
                    brightness=brightness,
                    shadow_strength=shadow,
                    noise_sigma=10,
                    tilt_px=5
                ))

    results = run_on_frames(synth, cfg)

    ok_results = [r for r in results if r.get("ok")]
    print(f"Total: {len(results)} | OK: {len(ok_results)} | Reject: {len(results)-len(ok_results)}")

    if ok_results:
        gaps = np.array([r["gap_px"] for r in ok_results], dtype=np.float32)
        print(f"gap_px median={np.median(gaps):.2f}, mean={np.mean(gaps):.2f}, std={np.std(gaps):.2f}")

    # Print a few examples
    for r in results[:5]:
        print(r)