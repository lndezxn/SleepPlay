from pathlib import Path

import cv2

from sleepplay.config import PreprocessConfig
from sleepplay.preprocess import preprocess_video
from video_helpers import write_test_video


def test_preprocess_video_writes_configured_fps_and_height(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "preprocessed.mp4"
    write_test_video(
        input_path,
        fps=4.0,
        size=(32, 24),
        values=(0, 0, 0, 0, 80, 80, 80, 80),
    )

    result_path = preprocess_video(
        input_path,
        PreprocessConfig(
            enabled=True,
            output=output_path,
            fps=1.0,
            height=12,
            video_codec="libx264",
            preset="veryfast",
            crf=28,
            pixel_format="yuv420p",
            overwrite=True,
        ),
    )

    capture = cv2.VideoCapture(str(result_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 1
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 12
    finally:
        capture.release()


def test_preprocess_disabled_returns_input_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "preprocessed.mp4"
    write_test_video(input_path)

    result_path = preprocess_video(
        input_path,
        PreprocessConfig(
            enabled=False,
            output=output_path,
            fps=1.0,
            height=12,
            video_codec="libx264",
            preset="veryfast",
            crf=28,
            pixel_format="yuv420p",
            overwrite=True,
        ),
    )

    assert result_path == input_path
    assert not output_path.exists()
