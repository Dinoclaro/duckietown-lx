import cv2
import numpy as np

lower_hsv = np.array([0, 60, 60])
upper_hsv = np.array([50, 255, 255])


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Returns a 2D array"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # This creates a binary mask where pixels in the range are set to white and outside range are set to black 
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask