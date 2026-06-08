import subprocess
from pathlib import Path

import imageio_ffmpeg

from sleepplay.config import PreprocessConfig


def preprocess_video(input_path: Path, config: PreprocessConfig) -> Path:
    if not config.enabled:
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if config.fps <= 0.0:
        raise ValueError("preprocess fps must be positive.")
    if config.height <= 0:
        raise ValueError("preprocess height must be positive.")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    overwrite_flag = "-y" if config.overwrite else "-n"
    video_filter = f"fps={config.fps},scale=-2:{config.height}"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        overwrite_flag,
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
        str(config.output),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed:\n{result.stderr}")

    return config.output
