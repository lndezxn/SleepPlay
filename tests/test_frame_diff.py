import numpy as np

from sleepplay.scores.frame_diff import FrameDiffScorer


def test_identical_frames_score_zero() -> None:
    scorer = FrameDiffScorer(analysis_width=8)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert scorer.score(frame, frame) == 0.0


def test_different_frames_score_positive() -> None:
    scorer = FrameDiffScorer(analysis_width=8)
    previous_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    current_frame = np.full((8, 8, 3), 50, dtype=np.uint8)

    assert scorer.score(previous_frame, current_frame) > 0.0
