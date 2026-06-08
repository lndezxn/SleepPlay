import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from sleepplay.config import PreprocessConfig
from sleepplay.progress import ProgressReporter, report_progress


def preprocess_video(
    input_path: Path,
    config: PreprocessConfig,
    progress_reporter: ProgressReporter | None = None,
) -> Path:
    if not config.enabled:
        report_progress(progress_reporter, "preprocess", 1.0, "Preprocessing skipped")
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if config.fps <= 0.0:
        raise ValueError("preprocess fps must be positive.")
    if config.height <= 0:
        raise ValueError("preprocess height must be positive.")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    report_progress(progress_reporter, "preprocess", 0.0, "Preprocessing video")
    overwrite_flag = "-y" if config.overwrite else "-n"
    video_filter = f"fps={config.fps},scale=-2:{config.height}"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        overwrite_flag,
        "-nostats",
        "-i",
        str(input_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        config.video_codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pixel_format,
        "-progress",
        "pipe:1",
        str(config.output),
    ]

    result = run_ffmpeg_preprocess(command, input_path, progress_reporter)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed:\n{result.stderr}")

    report_progress(progress_reporter, "preprocess", 1.0, "Preprocessing complete")
    return config.output


def run_ffmpeg_preprocess(
    command: list[str],
    input_path: Path,
    progress_reporter: ProgressReporter | None,
) -> subprocess.CompletedProcess[str]:
    if progress_reporter is None:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    duration_seconds = video_duration_seconds(input_path)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("ffmpeg progress pipes were not available.")

    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time":
            elapsed_seconds = parse_ffmpeg_time(value)
            report_progress(
                progress_reporter,
                "preprocess",
                elapsed_seconds / duration_seconds,
                "Preprocessing video",
            )

    stderr = process.stderr.read()
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.wait(),
        stdout="",
        stderr=stderr,
    )


def video_duration_seconds(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0.0 or frame_count <= 0:
            raise RuntimeError("Video must expose positive FPS and frame count metadata.")
        return frame_count / fps
    finally:
        capture.release()


def parse_ffmpeg_time(value: str) -> float:
    hours_text, minutes_text, seconds_text = value.split(":")
    return (
        int(hours_text) * 3600.0
        + int(minutes_text) * 60.0
        + float(seconds_text)
    )
