import os
import json
import time

import cv2
import depthai as dai
import numpy as np

# Same as your file (StereoDepth wants width multiple of 16; depth is aligned to RGB) :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
RGB_SIZE  = (640, 480)   # width, height
MONO_SIZE = (640, 480)

# Same visualization range you used (only affects the color preview, not the saved raw depth) :contentReference[oaicite:6]{index=6}
MIN_D = 150
MAX_D = 500

OUT_DIR = "captures"

def round_down_to_16(x: int) -> int:
    return (x // 16) * 16

def colorize_depth_mm(depth_mm: np.ndarray, min_d=MIN_D, max_d=MAX_D) -> np.ndarray:
    """uint16 depth in millimeters (0 = invalid) -> BGR uint8 colormap (for viewing)"""
    d = depth_mm.astype(np.float32)
    d[d <= 0] = np.nan
    d = np.clip(d, min_d, max_d)
    d = (d - min_d) * (255.0 / (max_d - min_d))
    d = np.nan_to_num(d, nan=0.0).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

os.makedirs(OUT_DIR, exist_ok=True)

with dai.Pipeline() as pipeline:
    # Cameras :contentReference[oaicite:7]{index=7}
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    cam_l   = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_B)
    cam_r   = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_C)

    # Stereo depth (same settings as your file) :contentReference[oaicite:8]{index=8}
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    stereo.setSubpixel(True)

    # Align depth to RGB camera (CAM_A) :contentReference[oaicite:9]{index=9}
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Force StereoDepth output size (width must be multiple of 16) :contentReference[oaicite:10]{index=10}
    out_w = round_down_to_16(RGB_SIZE[0])
    out_h = round_down_to_16(RGB_SIZE[1])
    stereo.setOutputSize(out_w, out_h)

    # Request outputs :contentReference[oaicite:11]{index=11}
    rgb_out   = cam_rgb.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p)
    left_out  = cam_l.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8)
    right_out = cam_r.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8)

    # Link stereo inputs :contentReference[oaicite:12]{index=12}
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    # Host queues :contentReference[oaicite:13]{index=13}
    q_rgb   = rgb_out.createOutputQueue()
    q_depth = stereo.depth.createOutputQueue()

    pipeline.start()

    last_rgb = None
    last_depth_mm = None
    last_depth_vis = None

    print("Press 's' to save one RGB + depth snapshot. Press 'q' to quit.")

    while pipeline.isRunning():
        m_rgb = q_rgb.tryGet()
        if m_rgb is not None:
            last_rgb = m_rgb.getCvFrame()

        m_depth = q_depth.tryGet()
        if m_depth is not None:
            last_depth_mm = m_depth.getFrame()  # uint16 depth in mm :contentReference[oaicite:14]{index=14}
            last_depth_vis = colorize_depth_mm(last_depth_mm)

        if last_rgb is not None and last_depth_vis is not None:
            # Depth may not match RGB exactly; resize for display only (raw depth saved unmodified)
            disp_depth = last_depth_vis
            if disp_depth.shape[:2] != last_rgb.shape[:2]:
                disp_depth = cv2.resize(
                    disp_depth, (last_rgb.shape[1], last_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            combined = np.hstack([last_rgb, disp_depth])
            cv2.imshow("RGB | Depth (preview)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("s") and last_rgb is not None and last_depth_mm is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")

            rgb_path       = os.path.join(OUT_DIR, f"rgb_{ts}.png")
            depth_raw_path = os.path.join(OUT_DIR, f"depth_mm_{ts}.png")
            depth_vis_path = os.path.join(OUT_DIR, f"depth_vis_{ts}.png")
            meta_path      = os.path.join(OUT_DIR, f"meta_{ts}.json")

            # Save RGB (8-bit, 3-channel)
            cv2.imwrite(rgb_path, last_rgb)

            # Save raw depth as 16-bit PNG (single channel). Values are millimeters; 0 = invalid.
            # IMPORTANT: keep dtype uint16 so it stays lossless.
            depth_u16 = last_depth_mm.astype(np.uint16, copy=False)
            cv2.imwrite(depth_raw_path, depth_u16)

            # Save a colorized depth preview (8-bit, 3-channel) for quick viewing
            cv2.imwrite(depth_vis_path, last_depth_vis)

            meta = {
                "timestamp": ts,
                "rgb_file": os.path.basename(rgb_path),
                "depth_raw_file": os.path.basename(depth_raw_path),
                "depth_vis_file": os.path.basename(depth_vis_path),
                "rgb_size_requested": list(RGB_SIZE),
                "mono_size_requested": list(MONO_SIZE),
                "stereo_output_size": [out_w, out_h],
                "depth_units": "millimeters",
                "depth_dtype": str(depth_u16.dtype),
                "depth_invalid_value": 0,
                "depth_aligned_to": "CAM_A (RGB)",
                "preview_colormap": "TURBO",
                "preview_range_mm": [MIN_D, MAX_D],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print("Saved:")
            print(" ", rgb_path)
            print(" ", depth_raw_path)
            print(" ", depth_vis_path)
            print(" ", meta_path)
            break

cv2.destroyAllWindows()

