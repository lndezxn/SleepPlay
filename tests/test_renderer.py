from pathlib import Path

import cv2
import numpy as np

from sleepplay.config import RenderConfig, RenderOverlayConfig
from sleepplay.renderer import (
    SourceFrameReader,
    build_replay_segments,
    draw_speed_overlay,
    render_replay_video,
)
from sleepplay.timeline import Timeline, TimelineRecord, read_timeline, write_timeline
from video_helpers import write_test_video


def test_read_timeline_loads_written_json(tmp_path: Path) -> None:
    timeline_path = tmp_path / "timeline.json"
    timeline = Timeline(
        video=str(tmp_path / "input.mp4"),
        render_video=str(tmp_path / "input.mp4"),
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=0.0, replay_speed=2.0),
            TimelineRecord(time=1.0, score=10.0, replay_speed=1.0),
        ],
    )

    write_timeline(timeline_path, timeline)

    assert read_timeline(timeline_path) == timeline


def test_build_replay_segments_uses_speed_and_final_interval() -> None:
    timeline = Timeline(
        video="input.mp4",
        render_video="input.mp4",
        frame_interval_seconds=1.0,
        records=[
            TimelineRecord(time=0.0, score=0.0, replay_speed=2.0),
            TimelineRecord(time=1.0, score=10.0, replay_speed=1.0),
        ],
    )

    segments = build_replay_segments(timeline, output_fps=30.0)

    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert segments[0].output_frame_count == 15
    assert segments[1].start == 1.0
    assert segments[1].end == 2.0
    assert segments[1].output_frame_count == 30


def test_draw_speed_overlay_changes_upper_right_pixels() -> None:
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    overlayed = draw_speed_overlay(
        frame,
        replay_speed=4.0,
        config=RenderOverlayConfig(
            enabled=True,
            margin=8,
            font_scale=0.6,
            thickness=2,
        ),
    )

    assert not np.array_equal(frame[:, 72:], overlayed[:, 72:])


def test_render_replay_video_writes_mp4(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    timeline_path = tmp_path / "timeline.json"
    output_path = tmp_path / "replay.mp4"
    write_test_video(
        video_path,
        fps=4.0,
        size=(64, 48),
        values=(0, 20, 40, 60, 80, 100, 120, 140),
    )
    write_timeline(
        timeline_path,
        Timeline(
            video=str(video_path),
            render_video=str(video_path),
            frame_interval_seconds=1.0,
            records=[
                TimelineRecord(time=0.0, score=0.0, replay_speed=2.0),
                TimelineRecord(time=1.0, score=10.0, replay_speed=1.0),
            ],
        ),
    )

    render_replay_video(
        RenderConfig(
            timeline_json=timeline_path,
            output_video=output_path,
            source="preprocessed",
            source_video=tmp_path / "render_source.mp4",
            source_fps=2.0,
            video_codec="libx264",
            quality=7,
            overlay=RenderOverlayConfig(
                enabled=True,
                margin=8,
                font_scale=0.5,
                thickness=1,
            ),
        )
    )

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 4
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        capture.release()


def test_source_frame_reader_skips_forward_without_seeking() -> None:
    capture = FakeCapture(frame_count=5)
    reader = SourceFrameReader(capture, source_fps=1.0, frame_count=5)

    reader.read(0.0)
    frame = reader.read(2.0)

    assert capture.set_calls == []
    assert capture.grab_calls == 1
    assert int(frame[0, 0, 0]) == 2


class FakeCapture:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.position = 0
        self.set_calls: list[int] = []
        self.grab_calls = 0

    def set(self, property_id: int, value: int) -> bool:
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.set_calls.append(value)
        self.position = value
        return True

    def grab(self) -> bool:
        if self.position >= self.frame_count:
            return False
        self.grab_calls += 1
        self.position += 1
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        if self.position >= self.frame_count:
            return False, np.zeros((1, 1, 3), dtype=np.uint8)
        frame = np.full((1, 1, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame
