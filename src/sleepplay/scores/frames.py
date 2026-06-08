import cv2
import numpy as np


def prepare_grayscale(frame: np.ndarray, analysis_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = analysis_width / width
    analysis_height = round(height * scale)
    resized = cv2.resize(frame, (analysis_width, analysis_height))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
