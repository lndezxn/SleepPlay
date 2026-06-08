import json
from pathlib import Path

from config_helpers import write_runtime_config
from sleepplay.config import load_config
from sleepplay.pipeline import build_timeline
from sleepplay.timeline import write_timeline
from video_helpers import write_test_video


def test_pipeline_writes_time_score_and_replay_speed(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "timeline.json"
    config_path = tmp_path / "config.yaml"

    write_test_video(video_path)
    write_runtime_config(
        path=config_path,
        video_path=video_path,
        preprocessed_path=tmp_path / "preprocessed.mp4",
        timeline_path=output_path,
        replay_path=tmp_path / "replay.mp4",
    )

    config = load_config(config_path)
    timeline = build_timeline(config)
    write_timeline(config.output.json, timeline)

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["video"] == str(video_path)
    assert data["render_video"] == str(tmp_path / "render_source.mp4")
    assert data["frame_interval_seconds"] == 1.0
    assert len(data["records"]) == 3
    assert set(data["records"][0]) == {"time", "score", "replay_speed"}
    assert data["records"][0]["time"] == 0.0
    assert data["records"][0]["score"] == 0.0
    assert 1.0 <= data["records"][0]["replay_speed"] <= 16.0
    assert data["records"][1]["score"] > 0.0
