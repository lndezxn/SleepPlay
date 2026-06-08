from pathlib import Path

import cv2
import numpy as np


def write_test_video(
    path: Path,
    fps: float = 2.0,
    size: tuple[int, int] = (16, 16),
    values: tuple[int, ...] = (0, 0, 80, 80, 160, 160),
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video: {path}")

    width, height = size
    for value in values:
        frame = np.full((height, width, 3), value, dtype=np.uint8)
        writer.write(frame)

    writer.release()
