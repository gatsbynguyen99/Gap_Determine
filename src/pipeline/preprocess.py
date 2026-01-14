from typing import Tuple
import cv2
import numpy as np
from .config import PreprocessCfg
#Make the downstream edge/segmentation step more stable across lighting changes.


def preprocess_to_gray(frame_bgr: np.ndarray, cfg: PreprocessCfg) -> np.ndarray:
    #covert BGR to grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    #apply CLAHE
    #boots local contrast under uneven illumination or directional lighting
    #no mouting or ring light
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=tuple(cfg.clahe_grid),
    )
    gray = clahe.apply(gray)

    #apply Gaussian blur
    #reduce sensor noise so edges dont become unstable and spiky

    k = cfg.blur_ksize
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return gray
