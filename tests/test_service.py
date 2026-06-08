from pathlib import Path

import cv2

from config_helpers import write_runtime_config
from sleepplay.config import load_config
from sleepplay.progress import ProgressUpdate
from sleepplay.service import config_for_job, process_replay
from video_helpers import write_test_video


def test_config_for_job_only_overrides_runtime_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_runtime_config(
        path=config_path,
        video_path=tmp_path / "base_input.mp4",
        preprocessed_path=tmp_path / "base_preprocessed.mp4",
        timeline_path=tmp_path / "base_timeline.json",
        replay_path=tmp_path / "base_replay.mp4",
    )
    config = load_config(config_path)

    job_config = config_for_job(config, tmp_path / "jobs/1/input.mp4", tmp_path / "jobs/1")

    assert job_config.video.input == tmp_path / "jobs/1/input.mp4"
    assert job_config.preprocess.output == tmp_path / "jobs/1/preprocessed.mp4"
    assert job_config.output.json == tmp_path / "jobs/1/timeline.json"
    assert job_config.render.timeline_json == tmp_path / "jobs/1/timeline.json"
    assert job_config.render.output_video == tmp_path / "jobs/1/replay.mp4"
    assert job_config.score == config.score
    assert job_config.speed == config.speed
    assert job_config.web == config.web


def test_process_replay_writes_outputs_and_reports_progress(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    config_path = tmp_path / "config.yaml"
    write_test_video(video_path, fps=4.0, size=(64, 48), values=(0, 40, 80, 120))
    write_runtime_config(
        path=config_path,
        video_path=video_path,
        preprocessed_path=tmp_path / "preprocessed.mp4",
        timeline_path=tmp_path / "timeline.json",
        replay_path=tmp_path / "replay.mp4",
    )
    config = load_config(config_path)
    updates: list[ProgressUpdate] = []

    result = process_replay(config, updates.append)

    assert result.timeline_path.exists()
    assert result.replay_path.exists()
    assert {update.stage for update in updates} >= {"preprocess", "timeline", "render"}
    assert updates[-1].progress == 1.0

    capture = cv2.VideoCapture(str(result.replay_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        capture.release()
