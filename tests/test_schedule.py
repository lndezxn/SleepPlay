import pytest

from sleepplay.schedule import (
    build_replay_scale,
    replay_time_for_source_time,
    source_time_for_replay_time,
)
from sleepplay.timeline import Timeline, TimelineRecord


def test_build_replay_scale_uses_final_frame_interval() -> None:
    timeline = Timeline(
        video="input.mp4",
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=1.0, replay_speed=2.0),
            TimelineRecord(time=1.0, score=4.0, replay_speed=1.0),
        ],
    )

    segments = build_replay_scale(timeline, output_fps=30.0)

    assert segments[0].source_start == 0.0
    assert segments[0].source_end == 1.0
    assert segments[0].replay_start == 0.0
    assert segments[0].replay_end == 0.5
    assert segments[1].source_start == 1.0
    assert segments[1].source_end == 2.0
    assert segments[1].replay_start == 0.5
    assert segments[1].replay_end == 1.5


def test_source_and_replay_time_mapping_round_trips() -> None:
    timeline = Timeline(
        video="input.mp4",
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=1.0, replay_speed=2.0),
            TimelineRecord(time=1.0, score=4.0, replay_speed=1.0),
        ],
    )
    segments = build_replay_scale(timeline, output_fps=30.0)

    replay_time = replay_time_for_source_time(segments, 0.4)
    source_time = source_time_for_replay_time(segments, replay_time)

    assert replay_time == 0.2
    assert source_time == 0.4


def test_build_replay_scale_uses_actual_output_frame_duration() -> None:
    timeline = Timeline(
        video="input.mp4",
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=1.0, replay_speed=60.0),
        ],
    )

    segments = build_replay_scale(timeline, output_fps=30.0)

    assert segments[0].output_frame_count == 1
    assert segments[0].replay_end == pytest.approx(1.0 / 30.0)
    assert segments[0].replay_end != pytest.approx(1.0 / 60.0)


def test_replay_boundary_uses_next_segment() -> None:
    timeline = Timeline(
        video="input.mp4",
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=1.0, replay_speed=2.0),
            TimelineRecord(time=1.0, score=4.0, replay_speed=1.0),
        ],
    )
    segments = build_replay_scale(timeline, output_fps=30.0)

    assert source_time_for_replay_time(segments, 0.5) == 1.0
