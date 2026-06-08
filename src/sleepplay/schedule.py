from dataclasses import dataclass
from math import floor

from sleepplay.renderer import build_replay_segments
from sleepplay.timeline import Timeline


@dataclass(frozen=True)
class ReplayScaleSegment:
    source_start: float
    source_end: float
    replay_start: float
    replay_end: float
    score: float
    replay_speed: float
    output_frame_count: int
    output_fps: float


def build_replay_scale(timeline: Timeline, output_fps: float) -> list[ReplayScaleSegment]:
    if output_fps <= 0.0:
        raise ValueError("output_fps must be positive.")
    if timeline.frame_interval_seconds <= 0.0:
        raise ValueError("frame_interval_seconds must be positive.")
    if not timeline.records:
        raise ValueError("timeline must contain at least one record.")

    render_segments = build_replay_segments(timeline, output_fps)
    segments: list[ReplayScaleSegment] = []
    replay_start = 0.0
    for record, render_segment in zip(timeline.records, render_segments, strict=True):
        replay_duration = render_segment.output_frame_count / output_fps
        replay_end = replay_start + replay_duration
        segments.append(
            ReplayScaleSegment(
                source_start=render_segment.start,
                source_end=render_segment.end,
                replay_start=replay_start,
                replay_end=replay_end,
                score=record.score,
                replay_speed=render_segment.replay_speed,
                output_frame_count=render_segment.output_frame_count,
                output_fps=output_fps,
            )
        )
        replay_start = replay_end

    return segments


def replay_time_for_source_time(
    segments: list[ReplayScaleSegment],
    source_time: float,
) -> float:
    segment = segment_for_source_time(segments, source_time)
    output_frame_index = round(
        ((source_time - segment.source_start) / segment.replay_speed)
        * segment.output_fps
    )
    output_frame_index = min(max(output_frame_index, 0), segment.output_frame_count - 1)
    return segment.replay_start + output_frame_index / segment.output_fps


def source_time_for_replay_time(
    segments: list[ReplayScaleSegment],
    replay_time: float,
) -> float:
    segment = segment_for_replay_time(segments, replay_time)
    output_frame_index = floor(
        (replay_time - segment.replay_start) * segment.output_fps
    )
    output_frame_index = min(max(output_frame_index, 0), segment.output_frame_count - 1)
    return min(
        segment.source_start
        + output_frame_index / segment.output_fps * segment.replay_speed,
        segment.source_end,
    )


def segment_for_source_time(
    segments: list[ReplayScaleSegment],
    source_time: float,
) -> ReplayScaleSegment:
    if not segments:
        raise ValueError("segments must not be empty.")
    clamped_time = min(max(source_time, segments[0].source_start), segments[-1].source_end)
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        if segment.source_start <= clamped_time < segment.source_end:
            return segment
        if is_last and segment.source_start <= clamped_time <= segment.source_end:
            return segment
    return segments[-1]


def segment_for_replay_time(
    segments: list[ReplayScaleSegment],
    replay_time: float,
) -> ReplayScaleSegment:
    if not segments:
        raise ValueError("segments must not be empty.")
    clamped_time = min(max(replay_time, segments[0].replay_start), segments[-1].replay_end)
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        if segment.replay_start <= clamped_time < segment.replay_end:
            return segment
        if is_last and segment.replay_start <= clamped_time <= segment.replay_end:
            return segment
    return segments[-1]
