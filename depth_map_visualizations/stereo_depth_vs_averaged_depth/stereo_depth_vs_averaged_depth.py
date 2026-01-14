import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# =========================
# Inputs (edit as needed)
# =========================
# Two depth-mm PNGs (uint16, millimeters; 0 = invalid)
DEPTH1_PNG_PATH = "depth_avg_mm_20260113_182842.png"
DEPTH2_PNG_PATH = "depth_mm_20260113_182842.png"

# Optional labels (leave as None to auto-use filename)
LABEL1 = None
LABEL2 = None

# Shared visualization range (mm) for BOTH depth maps
MIN_D = 200
MAX_D = 500

# If True: anything <MIN_D or >MAX_D becomes black (easier “band” comparison)
MASK_OUTSIDE_RANGE = False

# Output files
OUT_SIDE = "./depth_png_vs_png_side_by_side.png"
OUT_SIDE_XHAIR = "./depth_png_vs_png_side_by_side_with_crosshair.png"
OUT_ROW = "./cross_section_row_png_vs_png.png"
OUT_COL = "./cross_section_col_png_vs_png.png"


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


def plot_horizontal_cross_section_equal_aspect(y1: np.ndarray, y2: np.ndarray,
                                               label1: str, label2: str,
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
    ax.plot(x, y1, label=label1)
    ax.plot(x, y2, label=label2)
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

    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_vertical_cross_section_equal_aspect(x1: np.ndarray, x2: np.ndarray,
                                             label1: str, label2: str,
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
    ax.plot(x1, y, label=label1)
    ax.plot(x2, y, label=label2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # inverted y axis
    ax.set_title(f"Vertical cross-section at x={x0}")
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("y pixel")
    ax.legend()

    # Plot grid
    x_major, x_minor = 50, 10      # mm
    y_major, y_minor = 50, 10      # pixels

    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))

    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# =========================
# Load both PNGs (uint16 mm)
# =========================
d1_u16 = cv2.imread(DEPTH1_PNG_PATH, cv2.IMREAD_UNCHANGED)
if d1_u16 is None:
    raise RuntimeError(f"Failed to read depth PNG 1: {DEPTH1_PNG_PATH}")
if d1_u16.dtype != np.uint16:
    raise RuntimeError(f"Expected uint16 depth PNG 1. Got {d1_u16.dtype}")

H, W = d1_u16.shape

d2_u16 = cv2.imread(DEPTH2_PNG_PATH, cv2.IMREAD_UNCHANGED)
if d2_u16 is None:
    raise RuntimeError(f"Failed to read depth PNG 2: {DEPTH2_PNG_PATH}")
if d2_u16.dtype != np.uint16:
    raise RuntimeError(f"Expected uint16 depth PNG 2. Got {d2_u16.dtype}")

# Match size like your NPZ-resize path did (nearest so values stay “mm-like”)
if d2_u16.shape != (H, W):
    d2_u16 = cv2.resize(d2_u16, (W, H), interpolation=cv2.INTER_NEAREST)

label1 = LABEL1 if LABEL1 is not None else os.path.basename(DEPTH1_PNG_PATH)
label2 = LABEL2 if LABEL2 is not None else os.path.basename(DEPTH2_PNG_PATH)

# =========================
# Colorize with SAME LUT + SAME normalization
# =========================
v1 = depth_mm_to_color(d1_u16, MIN_D, MAX_D, MASK_OUTSIDE_RANGE)
v2 = depth_mm_to_color(d2_u16, MIN_D, MAX_D, MASK_OUTSIDE_RANGE)

combined = np.hstack([v1, v2])

cv2.putText(combined, label1, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(combined, label2, (W + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imwrite(OUT_SIDE, combined)

imshow_keep_aspect(
    cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
    f"Depth PNG vs Depth PNG (shared Turbo LUT + shared normalization)\n"
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
mm1 = d1_u16.astype(np.float32)
mm1[mm1 <= 0] = np.nan

mm2 = d2_u16.astype(np.float32)
mm2[mm2 <= 0] = np.nan

# Plot ranges
DEPTH_MIN = MIN_D - 50
DEPTH_MAX = MAX_D + 50

plot_horizontal_cross_section_equal_aspect(
    mm1[y0, :], mm2[y0, :],
    label1=label1, label2=label2,
    y0=y0, x_len=W,
    y_min=DEPTH_MIN, y_max=DEPTH_MAX,
    out_path=OUT_ROW
)

plot_vertical_cross_section_equal_aspect(
    mm1[:, x0], mm2[:, x0],
    label1=label1, label2=label2,
    x0=x0, y_len=H,
    x_min=DEPTH_MIN, x_max=DEPTH_MAX,
    out_path=OUT_COL
)

print("Wrote:")
print(" ", OUT_SIDE)
print(" ", OUT_SIDE_XHAIR)
print(" ", OUT_ROW)
print(" ", OUT_COL)
