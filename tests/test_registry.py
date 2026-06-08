import pytest
from pathlib import Path

from sleepplay.config import ScoreConfig, VideoConfig
from sleepplay.scores.frame_diff import FrameDiffScorer
from sleepplay.scores.gradient_diff import GradientDiffScorer
from sleepplay.scores.registry import create_scorer


def test_create_frame_diff_scorer() -> None:
    scorer = create_scorer(
        ScoreConfig(type="frame_diff"),
        VideoConfig(
            input=Path("input.mp4"),
            analysis_width=320,
        ),
    )

    assert isinstance(scorer, FrameDiffScorer)


def test_create_gradient_diff_scorer() -> None:
    scorer = create_scorer(
        ScoreConfig(type="gradient_diff"),
        VideoConfig(
            input=Path("input.mp4"),
            analysis_width=320,
        ),
    )

    assert isinstance(scorer, GradientDiffScorer)


def test_unknown_scorer_raises_error() -> None:
    with pytest.raises(ValueError, match="Unknown score algorithm"):
        create_scorer(
            ScoreConfig(type="missing"),
            VideoConfig(
                input=Path("input.mp4"),
                analysis_width=320,
            ),
        )
