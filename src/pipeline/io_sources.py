import os
import glob
import json
from typing import Iterator, Tuple, Dict

import cv2
import numpy as np


def iter_frames_from_folder(folder: str, pattern: str, max_frames: int = 0) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yields (frame_id, frame_bgr) from images in a folder.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if max_frames and max_frames > 0:
        paths = paths[:max_frames]

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield p, img


def iter_frames_from_camera(index: int = 0, max_frames: int = 0) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yields (frame_id, frame_bgr) from a webcam/USB camera.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"camera_frame_{count:06d}", frame
            count += 1
            if max_frames and count >= max_frames:
                break
    finally:
        cap.release()


def iter_captures_from_dir(captures_root: str, max_frames: int = 0) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict]]:
    """
    Recursively scans captures_root for files:
      rgb_<ts>.png
      depth_mm_<ts>.png
      meta_<ts>.json

    Supports subfolders like:
      captures/17 cm/, captures/21 cm/, ...

    Yields:
      (sample_id, rgb_bgr, depth_mm_uint16, meta_dict)

    sample_id includes folder name to avoid collisions.
    """
    rgb_paths = sorted(glob.glob(os.path.join(captures_root, "**", "rgb_*.png"), recursive=True))
    if max_frames and max_frames > 0:
        rgb_paths = rgb_paths[:max_frames]

    for rgb_path in rgb_paths:
        folder = os.path.basename(os.path.dirname(rgb_path))  # e.g. "17 cm"
        ts = os.path.basename(rgb_path).replace("rgb_", "").replace(".png", "")

        depth_path = os.path.join(os.path.dirname(rgb_path), f"depth_mm_{ts}.png")
        meta_path  = os.path.join(os.path.dirname(rgb_path), f"meta_{ts}.json")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # unique id: include distance folder + timestamp
        sample_id = f"{folder.replace(' ', '')}_{ts}"
        yield sample_id, rgb, depth, meta