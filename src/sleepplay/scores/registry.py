from sleepplay.config import ScoreConfig, VideoConfig
from sleepplay.scores.base import ScoreAlgorithm
from sleepplay.scores.frame_diff import FrameDiffScorer
from sleepplay.scores.gradient_diff import GradientDiffScorer


def create_scorer(config: ScoreConfig, video_config: VideoConfig) -> ScoreAlgorithm:
    if config.type == "frame_diff":
        return FrameDiffScorer(analysis_width=video_config.analysis_width)
    if config.type == "gradient_diff":
        return GradientDiffScorer(analysis_width=video_config.analysis_width)

    raise ValueError(f"Unknown score algorithm: {config.type}")
