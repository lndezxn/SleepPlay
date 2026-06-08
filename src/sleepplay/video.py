from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.progress import Progress


@dataclass(frozen=True)
class VideoFrame:
    time: float
    frame: np.ndarray


@dataclass(frozen=True)
class VideoFrames:
    frame_interval_seconds: float
    frames: list[VideoFrame]


def read_video_frames(path: Path) -> VideoFrames:
    if not path.exists():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0.0 or frame_count <= 0:
            raise RuntimeError("Video must expose positive FPS and frame count metadata.")

        frames: list[VideoFrame] = []
        with Progress() as progress:
            task = progress.add_task("Reading frames", total=frame_count)
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(VideoFrame(time=frame_index / fps, frame=frame))
                frame_index += 1
                progress.advance(task)

        if not frames:
            raise RuntimeError(f"Video did not contain readable frames: {path}")
        return VideoFrames(frame_interval_seconds=1.0 / fps, frames=frames)
    finally:
        capture.release()
