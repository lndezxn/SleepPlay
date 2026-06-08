from pathlib import Path

from config_helpers import write_runtime_config
from sleepplay.cli import main
from video_helpers import write_test_video


def test_run_command_writes_timeline_json(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "timeline.json"
    replay_path = tmp_path / "replay.mp4"
    config_path = tmp_path / "config.yaml"

    write_test_video(video_path)
    write_runtime_config(
        path=config_path,
        video_path=video_path,
        preprocessed_path=tmp_path / "preprocessed.mp4",
        timeline_path=output_path,
        replay_path=replay_path,
    )

    run_result = main(["run", "--config", str(config_path)])
    render_result = main(["render", "--config", str(config_path)])

    assert run_result == 0
    assert render_result == 0
    assert output_path.exists()
    assert replay_path.exists()
