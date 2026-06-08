from pathlib import Path

from sleepplay.config import load_config


def test_load_config_converts_yaml_to_dataclasses(tmp_path: Path) -> None:
    app_config_path = tmp_path / "app.yaml"
    score_config_path = tmp_path / "score.yaml"
    speed_config_path = tmp_path / "speed.yaml"
    config_path = tmp_path / "config.yaml"
    app_config_path.write_text(
        """
video:
  input: data/input.mp4
  analysis_width: 320
preprocess:
  enabled: true
  output: output/preprocessed.mp4
  fps: 1.0
  height: 320
  video_codec: libx264
  preset: veryfast
  crf: 28
  pixel_format: yuv420p
  overwrite: true
output:
  json: output/timeline.json
render:
  timeline_json: output/timeline.json
  output_video: output/replay.mp4
  fps: 30.0
  video_codec: libx264
  quality: 7
  overlay:
    enabled: true
    margin: 16
    font_scale: 0.8
    thickness: 2
web:
  host: 127.0.0.1
  port: 8000
  storage_root: data/web/jobs
  max_workers: 1
""",
        encoding="utf-8",
    )
    score_config_path.write_text(
        """
score:
  type: frame_diff
""",
        encoding="utf-8",
    )
    speed_config_path.write_text(
        """
speed:
  type: sensitive
  still_score: 1.0
  motion_score: 10.0
  min_speed: 1.0
  max_speed: 16.0
  sensitivity: 4.0
  smoothing_window: 3
""",
        encoding="utf-8",
    )
    config_path.write_text(
        """
includes:
  - app.yaml
  - score.yaml
  - speed.yaml

video:
  input: data/override.mp4
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.video.input == Path("data/override.mp4")
    assert config.video.analysis_width == 320
    assert config.preprocess.enabled is True
    assert config.preprocess.output == Path("output/preprocessed.mp4")
    assert config.preprocess.fps == 1.0
    assert config.preprocess.height == 320
    assert config.preprocess.video_codec == "libx264"
    assert config.preprocess.preset == "veryfast"
    assert config.preprocess.crf == 28
    assert config.preprocess.pixel_format == "yuv420p"
    assert config.preprocess.overwrite is True
    assert config.output.json == Path("output/timeline.json")
    assert config.render.timeline_json == Path("output/timeline.json")
    assert config.render.output_video == Path("output/replay.mp4")
    assert config.render.fps == 30.0
    assert config.render.video_codec == "libx264"
    assert config.render.quality == 7
    assert config.render.overlay.enabled is True
    assert config.render.overlay.margin == 16
    assert config.render.overlay.font_scale == 0.8
    assert config.render.overlay.thickness == 2
    assert config.web.host == "127.0.0.1"
    assert config.web.port == 8000
    assert config.web.storage_root == Path("data/web/jobs")
    assert config.web.max_workers == 1
    assert config.score.type == "frame_diff"
    assert config.score.params == {}
    assert config.speed.type == "sensitive"
    assert config.speed.still_score == 1.0
    assert config.speed.motion_score == 10.0
    assert config.speed.min_speed == 1.0
    assert config.speed.max_speed == 16.0
    assert config.speed.sensitivity == 4.0
    assert config.speed.smoothing_window == 3
