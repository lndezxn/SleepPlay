import cv2
import numpy as np

from sleepplay.scores.frames import prepare_grayscale


class FrameDiffScorer:
    def __init__(self, analysis_width: int) -> None:
        if analysis_width <= 0:
            raise ValueError("analysis_width must be positive.")
        self.analysis_width = analysis_width

    def score(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        previous_gray = prepare_grayscale(previous_frame, self.analysis_width)
        current_gray = prepare_grayscale(current_frame, self.analysis_width)
        diff = cv2.absdiff(previous_gray, current_gray)
        return float(np.mean(diff))
