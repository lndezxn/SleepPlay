from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sleepplay.config import SpeedConfig


@dataclass(frozen=True)
class SpeedContext:
    times: list[float]
    scores: list[float]
    frame_interval_seconds: float


class SpeedMapper(Protocol):
    def map_scores(self, context: SpeedContext) -> list[float]:
        """Return replay speeds for a score sequence."""


@dataclass(frozen=True)
class ScoreSeriesMapper:
    still_score: float
    motion_score: float
    min_speed: float
    max_speed: float
    pooling_window: int
    smoothing_window: int

    def __post_init__(self) -> None:
        if self.motion_score <= self.still_score:
            raise ValueError("motion_score must be greater than still_score.")
        if self.max_speed < self.min_speed:
            raise ValueError("max_speed must be greater than or equal to min_speed.")
        if self.pooling_window <= 0:
            raise ValueError("pooling_window must be positive.")
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive.")

    def map_scores(self, context: SpeedContext) -> list[float]:
        if len(context.times) != len(context.scores):
            raise ValueError("times and scores must have the same length.")
        if context.frame_interval_seconds <= 0.0:
            raise ValueError("frame_interval_seconds must be positive.")

        pooled_scores = max_pool_scores(context.scores, self.pooling_window)
        smoothed_scores = smooth_scores(pooled_scores, self.smoothing_window)
        return [self.map_score(score) for score in smoothed_scores]

    def map_score(self, score: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearSpeedMapper(ScoreSeriesMapper):

    def map_score(self, score: float) -> float:
        clamped_score = min(max(score, self.still_score), self.motion_score)
        motion_ratio = (clamped_score - self.still_score) / (
            self.motion_score - self.still_score
        )
        speed_range = self.max_speed - self.min_speed
        return self.max_speed - motion_ratio * speed_range


@dataclass(frozen=True)
class SensitiveSpeedMapper(ScoreSeriesMapper):
    sensitivity: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive.")

    def map_score(self, score: float) -> float:
        clamped_score = min(max(score, self.still_score), self.motion_score)
        motion_ratio = (clamped_score - self.still_score) / (
            self.motion_score - self.still_score
        )
        sensitive_ratio = 1.0 - (1.0 - motion_ratio) ** self.sensitivity
        speed_range = self.max_speed - self.min_speed
        return self.max_speed - sensitive_ratio * speed_range


def smooth_scores(scores: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive.")
    if window == 1 or not scores:
        return list(scores)

    values = np.array(scores, dtype=float)
    radius = window // 2
    smoothed_scores: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed_scores.append(float(np.mean(values[start:end])))
    return smoothed_scores


def max_pool_scores(scores: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive.")
    if window == 1 or not scores:
        return list(scores)

    values = np.array(scores, dtype=float)
    radius = window // 2
    pooled_scores: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        pooled_scores.append(float(np.max(values[start:end])))
    return pooled_scores


def create_speed_mapper(config: SpeedConfig) -> SpeedMapper:
    if config.type == "linear":
        return LinearSpeedMapper(
            still_score=config.still_score,
            motion_score=config.motion_score,
            min_speed=config.min_speed,
            max_speed=config.max_speed,
            pooling_window=config.pooling_window,
            smoothing_window=config.smoothing_window,
        )
    if config.type == "sensitive":
        return SensitiveSpeedMapper(
            still_score=config.still_score,
            motion_score=config.motion_score,
            min_speed=config.min_speed,
            max_speed=config.max_speed,
            pooling_window=config.pooling_window,
            smoothing_window=config.smoothing_window,
            sensitivity=config.sensitivity,
        )

    raise ValueError(f"Unknown speed mapper: {config.type}")
