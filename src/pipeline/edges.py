import cv2
import numpy as np
#edge function for reusable


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    #compute Scharr gradient in x and y
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    #using Scharr because it is stronger than Sobel for fine gradients
    #appropriate for boundary localization
    #magnitutude: squrt(gx^2 + gy^2)
    mag = cv2.magnitude(gx, gy)
    return mag

#alternative tool

def canny_edges(gray: np.ndarray, low: int = 40, high: int = 120) -> np.ndarray:
    return cv2.Canny(gray, low, high)
