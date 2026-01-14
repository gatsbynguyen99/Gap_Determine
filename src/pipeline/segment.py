from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from .config import MeasurementCfg, SegmentationCfg
from .measure import measure_gap_edges_from_mask

_ONNX_MODEL_CACHE = {}


def depth_foreground_mask(depth_mm: np.ndarray, band_mm: int = 40) -> np.ndarray:
    """
    Returns mask01 selecting pixels near the median depth (foreground plane).
    depth_mm: uint16 mm, invalid=0
    band_mm: allowed depth deviation around median
    """
    d = depth_mm.astype(np.int32)
    valid = d > 0
    if np.count_nonzero(valid) < 100:
        return np.ones_like(depth_mm, dtype=np.uint8)  # fallback: keep all

    z_med = int(np.median(d[valid]))
    fg = valid & (np.abs(d - z_med) <= band_mm)
    fg_u8 = (fg.astype(np.uint8) * 255)

    # cleanup
    k = np.ones((5, 5), np.uint8)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k, iterations=1)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k, iterations=1)
    return (fg_u8 == 255).astype(np.uint8)

# Connected-component selection (handles background pipes)
#in the senario that the background is not white but pipes.
#implement later
#Layer B
def select_gap_component(
    mask01, 
    expected_min, 
    expected_max, 
    center_prior=None, 
    depth_map=None
):
    """
    Purpose:
      When background pipes exist, the binary mask may contain multiple blobs.
      This function keeps ONLY the most plausible "gap" component.

    Inputs:
      mask01: binary mask in {0,1} (ROI-sized)
      expected_min / expected_max: plausible gap width range (in pixels) to reject wrong blobs
      center_prior: optional expected (cx, cy) in ROI coordinates; default is ROI center
      depth_map: optional depth aligned to ROI (same HxW). If provided, we prefer closer components.

    Output:
      selected_mask01: binary mask in {0,1} containing only the chosen component
    """
    H, W = mask01.shape
    mask_u8 = (mask01.astype(np.uint8) * 255)
    # connectedComponentsWithStats labels each connected blob and returns:
    # - labels: same shape as mask_u8, label id per pixel
    # - stats: bounding box + area per component
    # - centroids: (cx, cy) per component
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    #if only background exists, return original mask
    if num <= 1:
        return mask01
    
    #Choose expected center: ROI center or provided prior
    cx0, cy0 = (W/2.0, H/2.0) if center_prior is None else center_prior

    
    best_id = None
    best_score =  -1e18

    #iterate over components (skip label 0 which is background)
    for comp_id in range(1,num):
        x, y, w, h, area = stats[comp_id]
        cx, cy = centroids[comp_id]

        min_h = int(0.25 * H)   # require component at least 25% of ROI height
        if h < min_h:
            continue

        #Hard filterto reject obviously wrong components
        if area < 30: #then too small -> noise
            continue

        #width constraint: gap region should not be extremely wide/narrow
        #here w is component bbox width, which correlates with gap width
        if w < expected_min or w > expected_max:
            continue

        # base score (shape + location)
        dist2 = (cx - cx0)**2 + (cy - cy0)**2
        score = 2.0*h - 0.01*dist2 - 0.5*abs(w - (expected_min+expected_max)/2.0)

        # depth bonus: prefer closer component
        if depth_map is not None:
            comp_pixels = (labels == comp_id)
            d = depth_map[comp_pixels]
            d = d[d > 0]  # drop invalid=0
            if d.size > 0:
                score += -0.05*float(np.median(d))

        if score > best_score:
            best_score = score
            best_id = comp_id

    if best_id is None:
        return mask01

    return (labels == best_id).astype(np.uint8)
    
def _morph_cleanup(mask_u8: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup:
      OPEN removes isolated small blobs (noise)
      CLOSE fills small holes and connects small breaks
    """
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_u8


#Using Otsu thresholding as baseline placeholder
def segment_gap_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu thresholding:
      - Global threshold T automatically chosen
      - Produces binary mask
      - Works best when ROI histogram is bimodal
    """

    # Otsu chooses threshold automatically when thresh=0 and THRESH_OTSU flag used
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heuristic inversion:
    # If more than half the ROI is white, we assume "gap" is the smaller region and invert
    white_ratio = np.mean(mask == 255)
    if white_ratio > 0.5:
        mask = cv2.bitwise_not(mask)

    # Morph cleanup to stabilize mask
    mask = _morph_cleanup(mask)

    # Return 0/1 mask
    return (mask == 255).astype(np.uint8)

#Method 2: Adaptive thresholding applied when uneven lighting
def segment_gap_adaptive(gray: np.ndarray, block_size: int = 31, C: int = 3) -> np.ndarray:
    """
    Adaptive thresholding:
      - Computes a local threshold per neighborhood window
      - Better than Otsu when illumination varies across ROI

    Parameters:
      block_size (odd): neighborhood size
      C: subtract constant (tunes how strict threshold is)
    """
    # Ensure odd >= 3
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=int(C),
    )

    # Same inversion heuristic as Otsu
    if np.mean(mask == 255) > 0.5:
        mask = cv2.bitwise_not(mask)

    mask = _morph_cleanup(mask)
    return (mask == 255).astype(np.uint8)

#Method 3: Edge-guided with top-K + scoring (assuming background are fixed pipe)

def segment_gap_edge_guided_topk(
    gray: np.ndarray,
    meas_cfg: MeasurementCfg,
    top_k: int = 3,
    expected_gap_min: int = 5,
    expected_gap_max: int = 300,
    prior_edges: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Edge-guided fill-between-boundaries with top-K candidates:

    Core idea:
      - Find boundary edges using gradient magnitude (transitions)
      - For each scanline:
          * pick top-K candidate edges on left half and right half
          * score each pair (lx, rx)
          * choose the best pair and fill mask between them
      - Morphology to connect scanlines into a coherent region

      Applied with the background can be other pipes:
        Background pipes can produce strong edges.
      Top-K + scoring lets us choose the pair that matches expected gap range
      and stays consistent with prior frame (temporal prior).
    """
    H, W = gray.shape
    mid = W // 2

    # Compute gradient magnitude using Scharr (strong edge localization)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # Scanline positions
    y0 = int(H * meas_cfg.scanline_y_min)
    y1 = int(H * meas_cfg.scanline_y_max)
    y0 = max(0, min(y0, H - 1))
    y1 = max(y0 + 2, min(y1, H))
    ys = np.linspace(y0, y1 - 1, meas_cfg.n_scanlines).astype(int)

    mask = np.zeros((H, W), dtype=np.uint8)

    for yy in ys:
        row = mag[yy, :]

        left = row[:mid]
        right = row[mid:]

        kL = min(int(top_k), left.size)
        kR = min(int(top_k), right.size)

        # Get indices of top-K peaks (fast: argpartition)
        left_idx = np.argpartition(-left, kth=kL - 1)[:kL]
        right_idx = np.argpartition(-right, kth=kR - 1)[:kR] + mid

        best_pair = None
        best_score = -1e18

        for lx in left_idx:
            for rx in right_idx:
                gap = int(rx - lx)

                # Gap plausibility constraint
                if gap <= meas_cfg.min_gap_px:
                    continue

                # # Edge strength prefers strong boundaries
                strength = float(left[lx] + row[rx])

                # # Prefer gap near ROI center (optional)
                center_x = (lx + rx) / 2.0
                center_penalty = 0.01 * (center_x - (W / 2.0)) ** 2

                # # Temporal prior penalty (Layer D)
                prior_bonus = 0.0
                if prior_edges is not None:
                    plx, prx = prior_edges
                    # penalize large jumps
                    prior_bonus = -0.05 * ((lx - plx) ** 2 + (rx - prx) ** 2)

                # score = strength + prior_bonus - center_penalty
                target_gap = 0.5 * (expected_gap_min + expected_gap_max)

                # Soft penalty: prefer gaps near target, but do not drop scanlines
                range_penalty = 0.5 * abs(gap - target_gap)

                # Extra penalty if far outside the plausible range (still not a hard reject)
                if gap < expected_gap_min or gap > expected_gap_max:
                    range_penalty += 50.0

                score = strength + prior_bonus - center_penalty - range_penalty


                if score > best_score:
                    best_score = score
                    best_pair = (int(lx), int(rx))

        if best_pair is None:
            continue

        lx, rx = best_pair
        if rx - lx > meas_cfg.min_gap_px:
            mask[yy, lx:rx] = 255

    # Connect scanlines into a region
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    return (mask == 255).astype(np.uint8)


# def measure_gap_from_mask(mask01: np.ndarray, cfg: MeasurementCfg) -> Tuple[Optional[float], Dict]:
#     """
#     Scanline measurement on binary mask:
#     - For each scanline, find the longest contiguous run of 1s.
#     - Use that run length as the gap width in pixels.
#     - Return median across scanlines.
#     """
#     rh, rw = mask01.shape[:2]
#     y0 = int(rh * cfg.scanline_y_min)
#     y1 = int(rh * cfg.scanline_y_max)
#     y0 = max(0, min(y0, rh - 1))
#     y1 = max(0, min(y1, rh))

#     if y1 <= y0 + 1:
#         return None, {"valid_ratio": 0.0, "edge_strength": 0.0, "n_valid": 0}

#     ys = np.linspace(y0, y1 - 1, cfg.n_scanlines).astype(int)

#     widths = []
#     valid = 0

#     #for each scanline
#     #find all x indices where mask==1

#     for yy in ys:
#         row = mask01[yy, :]
#         idx = np.where(row > 0)[0]
#         if idx.size < 2:
#             continue

#         # Find longest contiguous segment in idx
#         splits = np.where(np.diff(idx) > 1)[0] + 1
#         groups = np.split(idx, splits)
#         longest = max(groups, key=lambda g: g.size)

#         #witdh = segment_end - segment_start + 1 
#         width = int(longest[-1] - longest[0] + 1)
#         if width <= cfg.min_gap_px:
#             continue

#         widths.append(width)
#         valid += 1

#     if valid == 0:
#         return None, {"valid_ratio": 0.0, "edge_strength": 0.0, "n_valid": 0}

#     widths = np.array(widths, dtype=np.float32)
#     info = {
#         "valid_ratio": float(valid / len(ys)),
#         "edge_strength": 999.0,  # not meaningful for mask method, not compute gradients here
#         "gap_px_mean": float(np.mean(widths)),
#         "gap_px_median": float(np.median(widths)),
#         "gap_px_std": float(np.std(widths)),
#         "n_valid": int(valid),
#     }
#     #take the median width across scanline
#     return float(np.median(widths)), info

#Add ML segmentation
#Open Neural Network Exchange
class OnnxSegModel:
    def __init__(self, onnx_path: str, input_size=(256, 256), thresh: float = 0.5):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.input_size = tuple(input_size)
        self.thresh = float(thresh)

    # def predict_mask01(self, gray_u8: np.ndarray) -> np.ndarray:
    #     """
    #     Input: gray ROI uint8 (H,W)
    #     Output: mask01 uint8 (H,W) in {0,1}
    #     """
    #     H, W = gray_u8.shape

    #     inp = cv2.resize(gray_u8, self.input_size, interpolation=cv2.INTER_AREA)
    #     inp = inp.astype(np.float32) / 255.0

    #     blob = inp[None, None, :, :]  # (1,1,h,w)
    #     self.net.setInput(blob)
    #     out = self.net.forward()

    #     prob = out.squeeze()  # (h,w)
    #     prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

    #     mask01 = (prob > self.thresh).astype(np.uint8)
    #     return mask01
    def predict_mask01(self, img: np.ndarray) -> np.ndarray:
        """
        Input:
        img can be:
            - grayscale uint8 (H,W)
            - BGR uint8 (H,W,3)
        Output:
        mask01 uint8 (H,W) in {0,1}
        """
        # Ensure 3 channels (model expects 3)
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        H, W = img_bgr.shape[:2]
        inW, inH = self.input_size[1], self.input_size[0]  # (H,W) stored, blobFromImage wants (W,H)

        blob = cv2.dnn.blobFromImage(
            img_bgr,
            scalefactor=1.0 / 255.0,
            size=(inW, inH),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,   # IMPORTANT: training used cv2.imread => BGR
            crop=False
        )

        self.net.setInput(blob)
        out = self.net.forward()

        # logits output is (inH, inW)
        logits = out.squeeze().astype(np.float32)
        prob = 1.0 / (1.0 + np.exp(-logits))

        # threshold at model resolution
        mask_small = (prob > self.thresh).astype(np.uint8)

        # resize mask back to original ROI size (W,H)
        mask01 = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask01




def segmentation_branch(
    gray: np.ndarray,
    seg_cfg: SegmentationCfg,
    meas_cfg: MeasurementCfg,
    temporal_state: Optional[Dict] = None,
    depth_map: Optional[np.ndarray] = None,
    img_bgr: Optional[np.ndarray] = None,
):
    """
    Returns:
      gap_px, info, mask01, temporal_state

    Notes:
      - temporal_state enables Layer D (camera stability).
      - depth_map enables depth-aware component selection (foreground preference).
      - Uses getattr(...) so you can add SegmentationCfg fields gradually without breaking.
    """
    if not seg_cfg.enabled:
        raise ValueError("Segmentation branch called but segmentation.enabled is false")

    method = seg_cfg.method

    # Read optional settings safely 
    component_filter_enabled = getattr(seg_cfg, "component_filter_enabled", True)
    expected_gap_px_min = int(getattr(seg_cfg, "expected_gap_px_min", 5))
    expected_gap_px_max = int(getattr(seg_cfg, "expected_gap_px_max", 300))
    adaptive_block_size = int(getattr(seg_cfg, "adaptive_block_size", 31))
    adaptive_C = int(getattr(seg_cfg, "adaptive_C", 3))
    edge_top_k = int(getattr(seg_cfg, "edge_top_k", 3))
    use_temporal_prior = bool(getattr(seg_cfg, "use_temporal_prior", False))

    prior_edges = None
    if use_temporal_prior and temporal_state is not None:
        prior_edges = temporal_state.get("prior_edges")

    # choose segmentation method
    if method == "otsu":
        mask01 = segment_gap_otsu(gray)

    elif method == "adaptive":
        mask01 = segment_gap_adaptive(gray, adaptive_block_size, adaptive_C)

    elif method == "edge_guided":
        mask01 = segment_gap_edge_guided_topk(
            gray,
            meas_cfg,
            top_k=edge_top_k,
            expected_gap_min=expected_gap_px_min,
            expected_gap_max=expected_gap_px_max,
            prior_edges=prior_edges,
        )
    elif method == "ml_onnx":
        model_path = getattr(seg_cfg, "model_path", None)
        if not model_path:
            raise ValueError("segmentation.method='ml_onnx' requires seg_cfg.model_path")

        input_size = getattr(seg_cfg, "onnx_input_size", (256, 256))
        thresh = float(getattr(seg_cfg, "onnx_thresh", 0.5))

        cache_key = (model_path, tuple(input_size), thresh)
        model = _ONNX_MODEL_CACHE.get(cache_key)
        if model is None:
            model = OnnxSegModel(model_path, input_size=input_size, thresh=thresh)
            _ONNX_MODEL_CACHE[cache_key] = model

        inp = img_bgr if img_bgr is not None else gray
        mask01 = model.predict_mask01(inp)


        mask01 = _morph_cleanup(mask01.astype(np.uint8) * 255)
        mask01 = (mask01 == 255).astype(np.uint8)



    else:
        raise NotImplementedError(f"Unknown segmentation method: {method}")

    # Layer B: keep only the most plausible gap component
    if depth_map is not None and method in ("otsu", "adaptive", "ml_onnx"):
        fg01 = depth_foreground_mask(depth_map, band_mm=60)  # can tune
        mask01 = (mask01 & fg01).astype(np.uint8)


    if component_filter_enabled:
        mask01 = select_gap_component(
            mask01,
            expected_min=expected_gap_px_min,
            expected_max=expected_gap_px_max,
            center_prior=None,
            depth_map=depth_map,
        )


    # measure gap from mask
    # gap_px, info = measure_gap_from_mask(mask01, meas_cfg)
    gap_px, info, edges = measure_gap_edges_from_mask(mask01, meas_cfg)

    # add endpoints into info so main.py can write them to CSV
    info["gap_x_left_roi"] = edges.get("x_left_med", None)
    info["gap_x_right_roi"] = edges.get("x_right_med", None)

    # optional: store lists for debugging (can be large)
    # info["gap_x_starts"] = edges.get("x_starts", [])
    # info["gap_x_ends"] = edges.get("x_ends", [])

    # Layer D: update temporal prior for next frame
    if temporal_state is None:
        temporal_state = {}

    if use_temporal_prior and gap_px is not None:
        # Estimate prior edges from a middle scanline
        yy = int(gray.shape[0] * 0.5)
        idx = np.where(mask01[yy] > 0)[0]
        if idx.size > 0:
            temporal_state["prior_edges"] = (int(idx[0]), int(idx[-1]))

    return gap_px, info, mask01, temporal_state

