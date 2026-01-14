import cv2
import depthai as dai
import numpy as np

# Pick sizes that are divisible by 16 (StereoDepth requirement)
RGB_SIZE  = (640, 480)   # width, height
MONO_SIZE = (640, 480)

# Depth visualization range (millimeters)
MIN_D = 150
MAX_D = 500

def round_down_to_16(x: int) -> int:
    return (x // 16) * 16

def colorize_depth_mm(depth_mm: np.ndarray, min_d=MIN_D, max_d=MAX_D) -> np.ndarray:
    """depth_mm: uint16 depth in millimeters (0 = invalid) -> BGR uint8 colormap"""
    d = depth_mm.astype(np.float32)
    invalid = (d <= 0)
    d[invalid] = np.nan

    d = np.clip(d, min_d, max_d)
    d = (d - min_d) * (255.0 / (max_d - min_d))
    d = np.nan_to_num(d, nan=0.0).astype(np.uint8)

    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

with dai.Pipeline() as pipeline:
    # Cameras
    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    cam_l   = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_B)
    cam_r   = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)      # Helps reject mismatches/occlusions
    stereo.setExtendedDisparity(True)   # Increase close-range converage
    stereo.setSubpixel(True)            # Increase depth precision

    # Align depth to RGB camera (CAM_A)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # IMPORTANT: force StereoDepth output size (width must be multiple of 16)
    out_w = round_down_to_16(RGB_SIZE[0])
    out_h = round_down_to_16(RGB_SIZE[1])
    stereo.setOutputSize(out_w, out_h)

    # Request outputs
    rgb_out   = cam_rgb.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p)
    left_out  = cam_l.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8)
    right_out = cam_r.requestOutput(MONO_SIZE, type=dai.ImgFrame.Type.GRAY8)

    # Link stereo inputs
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    # Host queues
    q_rgb   = rgb_out.createOutputQueue()
    q_depth = stereo.depth.createOutputQueue()

    pipeline.start()

    last_rgb = None
    last_depth_vis = None

    while pipeline.isRunning():
        m_rgb = q_rgb.tryGet()
        if m_rgb is not None:
            last_rgb = m_rgb.getCvFrame()

        m_depth = q_depth.tryGet()
        if m_depth is not None:
            depth_mm = m_depth.getFrame()  # uint16 depth in mm
            last_depth_vis = colorize_depth_mm(depth_mm)

        if last_rgb is not None and last_depth_vis is not None:
            # Resize depth to match RGB for display (they may not match exactly)
            if last_depth_vis.shape[:2] != last_rgb.shape[:2]:
                last_depth_vis = cv2.resize(
                    last_depth_vis, (last_rgb.shape[1], last_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            combined = np.hstack([last_rgb, last_depth_vis])
            cv2.imshow("RGB | Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

