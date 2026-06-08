import numpy as np

from sleepplay.scores.frame_diff import FrameDiffScorer
from sleepplay.scores.gradient_diff import GradientDiffScorer


def test_identical_frames_score_zero() -> None:
    scorer = GradientDiffScorer(analysis_width=16)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert scorer.score(frame, frame) == 0.0


def test_changed_edges_score_positive() -> None:
    scorer = GradientDiffScorer(analysis_width=16)
    previous_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    current_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    previous_frame[:, 4:8] = 255
    current_frame[:, 8:12] = 255

    assert scorer.score(previous_frame, current_frame) > 0.0


def test_uniform_brightness_shift_is_less_sensitive_than_pixel_diff() -> None:
    gradient_scorer = GradientDiffScorer(analysis_width=16)
    pixel_scorer = FrameDiffScorer(analysis_width=16)
    previous_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    current_frame = np.full((16, 16, 3), 50, dtype=np.uint8)

    gradient_score = gradient_scorer.score(previous_frame, current_frame)
    pixel_score = pixel_scorer.score(previous_frame, current_frame)

    assert gradient_score < pixel_score
