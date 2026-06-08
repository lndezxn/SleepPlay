from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from rich.progress import Progress

from sleepplay.config import RenderConfig, RenderOverlayConfig
from sleepplay.timeline import Timeline, read_timeline


@dataclass(frozen=True)
class ReplaySegment:
    start: float
    end: float
    replay_speed: float
    output_frame_count: int


def render_replay_video(config: RenderConfig) -> None:
    timeline = read_timeline(config.timeline_json)
    segments = build_replay_segments(timeline, output_fps=config.fps)
    source_path = Path(timeline.video)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if config.fps <= 0.0:
        raise ValueError("render fps must be positive.")

    config.output_video.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_fps <= 0.0 or source_frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Source video must expose valid metadata.")

        frame_reader = SourceFrameReader(capture, source_fps, source_frame_count)
        writer = imageio_ffmpeg.write_frames(
            str(config.output_video),
            size=(width, height),
            fps=config.fps,
            codec=config.video_codec,
            quality=config.quality,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
        )
        writer.send(None)

        try:
            total_frames = sum(segment.output_frame_count for segment in segments)
            with Progress() as progress:
                task = progress.add_task("Rendering replay", total=total_frames)
                for segment in segments:
                    for output_frame_index in range(segment.output_frame_count):
                        source_time = source_time_for_output_frame(
                            segment,
                            output_frame_index,
                            config.fps,
                        )
                        source_frame = frame_reader.read(source_time)
                        overlayed_frame = draw_speed_overlay(
                            source_frame,
                            segment.replay_speed,
                            config.overlay,
                        )
                        rgb_frame = cv2.cvtColor(overlayed_frame, cv2.COLOR_BGR2RGB)
                        writer.send(np.ascontiguousarray(rgb_frame))
                        progress.advance(task)
        finally:
            writer.close()
    finally:
        capture.release()


def build_replay_segments(timeline: Timeline, output_fps: float) -> list[ReplaySegment]:
    if output_fps <= 0.0:
        raise ValueError("output_fps must be positive.")
    if timeline.frame_interval_seconds <= 0.0:
        raise ValueError("frame_interval_seconds must be positive.")
    if not timeline.records:
        raise ValueError("timeline must contain at least one record.")

    segments: list[ReplaySegment] = []
    for index, record in enumerate(timeline.records):
        if record.replay_speed <= 0.0:
            raise ValueError("replay_speed must be positive.")

        if index + 1 < len(timeline.records):
            end = timeline.records[index + 1].time
        else:
            end = record.time + timeline.frame_interval_seconds

        source_duration = end - record.time
        if source_duration <= 0.0:
            raise ValueError("timeline record times must be increasing.")

        output_duration = source_duration / record.replay_speed
        output_frame_count = max(1, round(output_duration * output_fps))
        segments.append(
            ReplaySegment(
                start=record.time,
                end=end,
                replay_speed=record.replay_speed,
                output_frame_count=output_frame_count,
            )
        )

    return segments


def source_time_for_output_frame(
    segment: ReplaySegment,
    output_frame_index: int,
    output_fps: float,
) -> float:
    output_elapsed = output_frame_index / output_fps
    source_time = segment.start + output_elapsed * segment.replay_speed
    return min(source_time, segment.end)


def draw_speed_overlay(
    frame: np.ndarray,
    replay_speed: float,
    config: RenderOverlayConfig,
) -> np.ndarray:
    overlayed_frame = frame.copy()
    if not config.enabled:
        return overlayed_frame

    text = f"{replay_speed:.1f}x"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(
        text,
        font_face,
        config.font_scale,
        config.thickness,
    )
    text_width, text_height = text_size
    padding = max(4, config.margin // 3)
    frame_height, frame_width = overlayed_frame.shape[:2]
    text_x = max(config.margin, frame_width - config.margin - text_width)
    text_y = config.margin + text_height

    background_left = max(0, text_x - padding)
    background_top = max(0, text_y - text_height - padding)
    background_right = min(frame_width, text_x + text_width + padding)
    background_bottom = min(frame_height, text_y + baseline + padding)
    cv2.rectangle(
        overlayed_frame,
        (background_left, background_top),
        (background_right, background_bottom),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        overlayed_frame,
        text,
        (text_x, text_y),
        font_face,
        config.font_scale,
        (255, 255, 255),
        config.thickness,
        cv2.LINE_AA,
    )
    return overlayed_frame


class SourceFrameReader:
    def __init__(
        self,
        capture: cv2.VideoCapture,
        source_fps: float,
        frame_count: int,
    ) -> None:
        self.capture = capture
        self.source_fps = source_fps
        self.frame_count = frame_count
        self.cached_frame_index = -1
        self.cached_frame: np.ndarray | None = None

    def read(self, source_time: float) -> np.ndarray:
        target_frame_index = min(
            max(round(source_time * self.source_fps), 0),
            self.frame_count - 1,
        )
        if target_frame_index == self.cached_frame_index and self.cached_frame is not None:
            return self.cached_frame

        if target_frame_index != self.cached_frame_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)

        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Could not read source frame {target_frame_index}.")

        self.cached_frame_index = target_frame_index
        self.cached_frame = frame
        return frame
