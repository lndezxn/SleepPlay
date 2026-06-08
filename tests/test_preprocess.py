from pathlib import Path

import cv2

from sleepplay.config import PreprocessConfig
from sleepplay.preprocess import preprocess_video, preprocess_video_pair
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


def test_preprocess_video_pair_writes_analysis_and_render_sources(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    analysis_path = tmp_path / "analysis.mp4"
    render_source_path = tmp_path / "render_source.mp4"
    write_test_video(
        input_path,
        fps=4.0,
        size=(32, 24),
        values=(0, 20, 40, 60, 80, 100, 120, 140),
    )

    analysis_result, render_source_result = preprocess_video_pair(
        input_path,
        PreprocessConfig(
            enabled=True,
            output=analysis_path,
            fps=1.0,
            height=12,
            video_codec="libx264",
            preset="veryfast",
            crf=28,
            pixel_format="yuv420p",
            overwrite=True,
        ),
        PreprocessConfig(
            enabled=True,
            output=render_source_path,
            fps=2.0,
            height=12,
            video_codec="libx264",
            preset="veryfast",
            crf=28,
            pixel_format="yuv420p",
            overwrite=True,
        ),
    )

    assert analysis_result == analysis_path
    assert render_source_result == render_source_path
    assert video_fps(analysis_result) == 1
    assert video_fps(render_source_result) == 2
    assert video_height(analysis_result) == 12
    assert video_height(render_source_result) == 12


def video_fps(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        return round(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def video_height(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        return int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
