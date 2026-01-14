import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================
# Inputs (edit as needed)
# =========================
# Method 1: Stereo camera depth map (DepthAI / StereoDepth output)
STEREO_PNG_PATH = "../mockup_dataset/captures/depth_mm_20260113_025001.png"

# Method 2: ML-Depth-Pro depth output (your saved inference result)
ML_DEPTH_PRO_NPZ_PATH = "../mockup_dataset/captures/rgb_20260113_025001_depth.npz"

# Shared visualization range (mm) for BOTH methods
MIN_D = 100
MAX_D = 1500

# If True: anything <MIN_D or >MAX_D becomes black (easier “band” comparison)
MASK_OUTSIDE_RANGE = False

# Output files
OUT_SIDE = "./depth_side_by_side.png"
OUT_SIDE_XHAIR = "./depth_side_by_side_with_crosshair.png"
OUT_ROW = "./cross_section_row.png"
OUT_COL = "./cross_section_col.png"


# =========================
# Shared colormap (ONE LUT)
# =========================
_TURBO_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(256, 1),
    cv2.COLORMAP_TURBO
)[:, 0, :]  # (256, 3) BGR


def depth_mm_to_color(depth_mm_any, min_d=MIN_D, max_d=MAX_D, mask_outside=MASK_OUTSIDE_RANGE):
    """
    Convert a depth map to a color image using the SAME LUT and SAME normalization.
    depth_mm_any can be uint16 (mm, 0=invalid) or float mm (NaN/<=0 invalid).
    Returns BGR uint8.
    """
    d = depth_mm_any.astype(np.float32)

    invalid = (~np.isfinite(d)) | (d <= 0)
    if mask_outside:
        invalid = invalid | (d < min_d) | (d > max_d)

    d_clip = np.clip(d, min_d, max_d)
    idx = ((d_clip - min_d) * (255.0 / (max_d - min_d))).astype(np.uint8)

    out = _TURBO_LUT[idx]  # (H,W,3) BGR
    out[invalid] = (0, 0, 0)
    return out


def imshow_keep_aspect(rgb_img: np.ndarray, title: str, fig_w: float = 12.0):
    """
    Display an image with square pixels (aspect='equal') and a figure sized to match the image aspect ratio.
    """
    h, w = rgb_img.shape[:2]
    fig_h = fig_w * (h / w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb_img)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(pad=0)
    plt.show()


def squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    """
    Make sure the loaded depth is 2D.
    Accepts (H,W), (H,W,1), or (1,H,W).
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0, :, :]
    raise ValueError(f"Expected a 2D depth array; got shape={arr.shape}")


def plot_horizontal_cross_section_equal_aspect(y1: np.ndarray, y2: np.ndarray,
                                               y0: int, x_len: int,
                                               y_min: float, y_max: float,
                                               out_path: str):
    """
    Equal aspect for a line plot means: 1 unit in x == 1 unit in y in screen space.
    For horizontal cross-section: x is pixel index, y is depth (mm).
    We enforce ax.set_aspect('equal') AND choose figsize to match (y_range/x_range).
    """
    x = np.arange(x_len, dtype=np.float32)
    x_min, x_max = 0.0, float(x_len - 1)
    x_range = max(1.0, x_max - x_min)
    y_range = max(1.0, float(y_max - y_min))

    # Pick a reasonable base width then compute height to keep equal scaling.
    fig_w = 12.0
    fig_h = fig_w * (y_range / x_range)
    fig_h = max(3.0, min(12.0, fig_h))  # clamp to avoid extreme skinny/tall figures

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(x, y1, label="Stereo camera depth map")
    ax.plot(x, y2, label="ML-Depth-Pro")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Horizontal cross-section at y={y0}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("depth (mm)")
    ax.legend()

    # Plot grid
    x_major, x_minor = 50, 10      # pixels
    y_major, y_minor = 50, 10      # mm

    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))

    # draw both grids
    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_vertical_cross_section_equal_aspect(x1: np.ndarray, x2: np.ndarray,
                                             x0: int, y_len: int,
                                             x_min: float, x_max: float,
                                             out_path: str):
    """
    For vertical cross-section: x is depth (mm), y is pixel index.
    We enforce equal scaling by setting aspect='equal' and choosing figsize
    to match (y_range/x_range).
    """
    y = np.arange(y_len, dtype=np.float32)
    y_min, y_max = 0.0, float(y_len - 1)
    y_range = max(1.0, y_max - y_min)
    x_range = max(1.0, float(x_max - x_min))

    # Pick a reasonable base height then compute width to keep equal scaling.
    fig_h = 8.0
    fig_w = fig_h * (x_range / y_range)
    fig_w = max(3.0, min(12.0, fig_w))  # clamp

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(x1, y, label="Stereo camera depth map")
    ax.plot(x2, y, label="ML-Depth-Pro")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # inverted y axis
    ax.set_title(f"Vertical cross-section at x={x0}")
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("y pixel")
    ax.legend()

    # Plot grid
    x_major, x_minor = 50, 10      # pixels
    y_major, y_minor = 50, 10      # mm

    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))

    # draw both grids
    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# =========================
# Load Method 1 (Stereo camera depth map: PNG uint16 mm)
# =========================
stereo_depth_u16 = cv2.imread(STEREO_PNG_PATH, cv2.IMREAD_UNCHANGED)
if stereo_depth_u16 is None:
    raise RuntimeError(f"Failed to read stereo depth PNG: {STEREO_PNG_PATH}")
if stereo_depth_u16.dtype != np.uint16:
    raise RuntimeError(f"Expected uint16 stereo depth PNG. Got {stereo_depth_u16.dtype}")

H, W = stereo_depth_u16.shape


# =========================
# Load Method 2 (ML-Depth-Pro: NPZ -> mm)
# =========================
npz = np.load(ML_DEPTH_PRO_NPZ_PATH)

candidate_key = None
for k in ["depth_mm", "depth", "pred_depth", "prediction", "depth_map", "z", "d"]:
    if k in npz:
        candidate_key = k
        break
if candidate_key is None and len(npz.files) == 1:
    candidate_key = npz.files[0]
if candidate_key is None:
    raise KeyError(f"Couldn't find a depth key in NPZ. Keys={list(npz.files)}")

d = squeeze_to_2d(npz[candidate_key]).astype(np.float32)

valid = np.isfinite(d) & (d > 0)
med = float(np.nanmedian(d[valid])) if np.any(valid) else float("nan")
ml_depth_mm = d * 1000.0 if (np.isfinite(med) and med < 20.0) else d

if ml_depth_mm.shape != (H, W):
    ml_depth_mm = cv2.resize(ml_depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)


# =========================
# Colorize with SAME LUT + SAME normalization
# =========================
stereo_vis = depth_mm_to_color(stereo_depth_u16, MIN_D, MAX_D, MASK_OUTSIDE_RANGE)
mldp_vis = depth_mm_to_color(ml_depth_mm, MIN_D, MAX_D, MASK_OUTSIDE_RANGE)

combined = np.hstack([stereo_vis, mldp_vis])

cv2.putText(
    combined,
    f"Stereo camera depth map",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)
cv2.putText(
    combined,
    f"ML-Depth-Pro",
    (W + 10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.imwrite(OUT_SIDE, combined)

imshow_keep_aspect(
    cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
    f"Stereo vs ML-Depth-Pro (shared Turbo LUT + shared normalization)\n"
    f"Range: {MIN_D}-{MAX_D} mm"
)


# =========================
# Crosshair + cross-sections (EQUAL ASPECT)
# =========================
y0 = H // 2
x0 = W // 5 * 3

combined_x = combined.copy()
cv2.line(combined_x, (0, y0), (combined_x.shape[1] - 1, y0), (255, 255, 255), 1)
cv2.line(combined_x, (x0, 0), (x0, H - 1), (255, 255, 255), 1)
cv2.line(combined_x, (W + x0, 0), (W + x0, H - 1), (255, 255, 255), 1)

cv2.imwrite(OUT_SIDE_XHAIR, combined_x)
imshow_keep_aspect(cv2.cvtColor(combined_x, cv2.COLOR_BGR2RGB),
                   "Cross-section locations (white lines)")

# Cross-sections in mm (NaN for invalid)
stereo_mm = stereo_depth_u16.astype(np.float32)
stereo_mm[stereo_mm <= 0] = np.nan

mldp_mm = ml_depth_mm.astype(np.float32)
mldp_mm[~np.isfinite(mldp_mm) | (mldp_mm <= 0)] = np.nan

# Plot ranges
DEPTH_MIN = MIN_D - 50
DEPTH_MAX = MAX_D + 50

# Horizontal equal-aspect cross-section
plot_horizontal_cross_section_equal_aspect(
    stereo_mm[y0, :], mldp_mm[y0, :],
    y0=y0, x_len=W,
    y_min=DEPTH_MIN, y_max=DEPTH_MAX,
    out_path=OUT_ROW
)

# Vertical equal-aspect cross-section
plot_vertical_cross_section_equal_aspect(
    stereo_mm[:, x0], mldp_mm[:, x0],
    x0=x0, y_len=H,
    x_min=DEPTH_MIN, x_max=DEPTH_MAX,
    out_path=OUT_COL
)

print("Wrote:")
print(" ", OUT_SIDE)
print(" ", OUT_SIDE_XHAIR)
print(" ", OUT_ROW)
print(" ", OUT_COL)
print("ML-Depth-Pro NPZ keys:", list(npz.files))
print("ML-Depth-Pro key used:", candidate_key)
