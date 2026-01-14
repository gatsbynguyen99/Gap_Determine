import os
import glob
import json

import cv2
import numpy as np

CAPTURE_DIR = "captures"   # same folder used by the saver
MIN_D = 300                # mm (change to match your preference)
MAX_D = 5000               # mm

def colorize_depth_mm(depth_mm: np.ndarray, min_d=MIN_D, max_d=MAX_D) -> np.ndarray:
    """uint16 depth in millimeters (0 = invalid) -> BGR uint8 colormap"""
    d = depth_mm.astype(np.float32)
    invalid = (d <= 0)
    d[invalid] = np.nan

    d = np.clip(d, min_d, max_d)
    d = (d - min_d) * (255.0 / (max_d - min_d))
    d = np.nan_to_num(d, nan=0.0).astype(np.uint8)

    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

def newest_capture_paths(capture_dir: str):
    depth_files = sorted(glob.glob(os.path.join(capture_dir, "depth_mm_*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No depth_mm_*.png found in: {capture_dir}")

    depth_path = depth_files[-1]
    ts = os.path.basename(depth_path).replace("depth_mm_", "").replace(".png", "")

    rgb_path  = os.path.join(capture_dir, f"rgb_{ts}.png")
    meta_path = os.path.join(capture_dir, f"meta_{ts}.json")

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"Found depth but missing rgb file: {rgb_path}")

    return rgb_path, depth_path, (meta_path if os.path.exists(meta_path) else None), ts

rgb_path, depth_path, meta_path, ts = newest_capture_paths(CAPTURE_DIR)

# Load RGB (OpenCV reads as BGR uint8)
rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
if rgb is None:
    raise RuntimeError(f"Failed to read RGB image: {rgb_path}")

# Load depth EXACTLY as saved: uint16 single-channel
depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if depth_mm is None:
    raise RuntimeError(f"Failed to read depth image: {depth_path}")
if depth_mm.dtype != np.uint16:
    raise RuntimeError(f"Depth image isn't uint16. Did you use IMREAD_UNCHANGED? dtype={depth_mm.dtype}")

# Optionally pull visualization range from metadata if present
if meta_path:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "preview_range_mm" in meta:
        MIN_D, MAX_D = meta["preview_range_mm"]

depth_vis = colorize_depth_mm(depth_mm, MIN_D, MAX_D)

# Resize depth visualization to match RGB for side-by-side display
if depth_vis.shape[:2] != rgb.shape[:2]:
    depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

combined = np.hstack([rgb, depth_vis])

print("Loaded capture:", ts)
print("RGB:", rgb_path)
print("Depth:", depth_path)
if meta_path:
    print("Meta:", meta_path)
print("Depth format: uint16 PNG, millimeters, 0 = invalid")

# Click-to-query depth helper
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # If you click on the right half, map into depth image coordinates
    w = rgb.shape[1]
    if x >= w:
        dx = x - w
        dy = y
        if 0 <= dy < depth_mm.shape[0] and 0 <= dx < depth_mm.shape[1]:
            d = int(depth_mm[dy, dx])
            if d == 0:
                print(f"Depth at (x={dx}, y={dy}): invalid (0)")
            else:
                print(f"Depth at (x={dx}, y={dy}): {d} mm ({d/1000.0:.3f} m)")

cv2.namedWindow("RGB | Depth (replay)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("RGB | Depth (replay)", on_mouse)
cv2.imshow("RGB | Depth (replay)", combined)

print("Controls: press 'q' or ESC to quit.")
while True:
    k = cv2.waitKey(0) & 0xFF
    if k in (ord("q"), 27):
        break

cv2.destroyAllWindows()

