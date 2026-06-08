import cv2
import numpy as np

from sleepplay.scores.frames import prepare_grayscale


class GradientDiffScorer:
    def __init__(self, analysis_width: int) -> None:
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive.")
        self.analysis_width = analysis_width

    def score(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        previous_gradient = self._gradient_magnitude(previous_frame)
        current_gradient = self._gradient_magnitude(current_frame)
        diff = cv2.absdiff(previous_gradient, current_gradient)
        return float(np.mean(diff))

    def _gradient_magnitude(self, frame: np.ndarray) -> np.ndarray:
        gray = prepare_grayscale(frame, self.analysis_width)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gradient_x, gradient_y)
