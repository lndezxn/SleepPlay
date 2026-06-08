import json
from pathlib import Path

from fastapi.testclient import TestClient

from config_helpers import write_runtime_config
from sleepplay.config import AppConfig, load_config
from sleepplay.progress import ProgressReporter, ProgressUpdate
from sleepplay.service import ProcessingResult
from sleepplay.timeline import Timeline, TimelineRecord, write_timeline
from sleepplay.web import create_app


def test_web_index_contains_upload_video_and_timelines(tmp_path: Path) -> None:
    config = web_test_config(tmp_path)
    client = TestClient(create_app(config, fake_processor))

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="video-input"' in response.text
    assert 'id="theme-button"' in response.text
    assert 'id="theme-icon"' in response.text
    assert 'aria-label="Switch to dark mode"' in response.text
    assert 'data-theme' in response.text
    assert ':root[data-theme="dark"]' in response.text
    assert 'localStorage.getItem("sleepplay.theme")' in response.text
    assert 'localStorage.setItem(themeStorageKey, theme)' in response.text
    assert 'prefers-color-scheme: dark' in response.text
    assert "setTheme" in response.text
    assert "syncThemeButton" in response.text
    assert 'id="replay-video"' in response.text
    assert 'id="reset-button"' in response.text
    assert 'id="back-button"' in response.text
    assert 'id="play-button"' in response.text
    assert 'id="forward-button"' in response.text
    assert 'id="fullscreen-button"' in response.text
    assert 'aria-label="Reset"' in response.text
    assert 'aria-label="Back 10 seconds"' in response.text
    assert 'aria-label="Play"' in response.text
    assert 'aria-label="Forward 10 seconds"' in response.text
    assert 'aria-label="Fullscreen"' in response.text
    assert "object-fit: cover" in response.text
    assert ".result:fullscreen" in response.text
    assert ".result:fullscreen.fullscreen-ui-visible .timeline-stack" in response.text
    assert ".result:fullscreen svg.timeline" in response.text
    assert "height: 30px" in response.text
    assert "requestFullscreen" in response.text
    assert "exitFullscreen" in response.text
    assert "fullscreenHideTimer" in response.text
    assert "showFullscreenChrome" in response.text
    assert "hideFullscreenChrome" in response.text
    assert "fullscreen-ui-visible" in response.text
    assert '<svg id="source-timeline"' in response.text
    assert '<svg id="replay-timeline"' in response.text
    assert 'id="source-time"' in response.text
    assert 'id="replay-time"' in response.text
    assert 'id="source-preview"' in response.text
    assert 'id="replay-preview"' in response.text
    assert "Processing Settings" in response.text
    assert 'name="score_type"' in response.text
    assert 'name="speed_type"' in response.text
    assert 'name="analysis_width"' in response.text
    assert 'name="preprocess_fps"' in response.text
    assert 'name="render_fps"' in response.text
    assert "hydrateSettings" in response.text
    assert "new FormData(form)" in response.text
    assert "__DEFAULT_SETTINGS__" not in response.text
    assert 'data-stage="upload"' in response.text
    assert 'data-stage="preprocess"' in response.text
    assert 'data-stage="timeline"' in response.text
    assert 'data-stage="render"' in response.text
    assert "updateStageProgress" in response.text
    assert "resetStageProgress" in response.text
    assert 'id="progress-fill"' not in response.text
    assert "--red: #c8324a" in response.text
    assert "var(--score)" in response.text
    assert 'event.code === "Space"' in response.text
    assert "keyboardSeekSeconds = 1" in response.text
    assert 'event.code === "ArrowLeft"' in response.text
    assert 'event.code === "ArrowRight"' in response.text
    assert "previewCursorElement" in response.text
    assert "scoreForSourceTime" in response.text
    assert "formatScore" in response.text
    assert "score ${formatScore(score)}" in response.text
    assert 'preserveAspectRatio", "none"' in response.text
    assert "#287f9f" not in response.text
    assert "buildStepPath" not in response.text


def test_upload_creates_job_and_streams_progress(tmp_path: Path) -> None:
    config = web_test_config(tmp_path)
    client = TestClient(create_app(config, fake_processor))

    response = client.post(
        "/jobs",
        data={
            "score_type": "gradient_diff",
            "analysis_width": "64",
            "preprocess_fps": "2.0",
            "preprocess_height": "48",
            "render_fps": "12.0",
            "speed_type": "linear",
            "still_score": "0.5",
            "motion_score": "8.5",
            "min_speed": "1.5",
            "max_speed": "20.0",
            "sensitivity": "2.5",
            "smoothing_window": "5",
        },
        files={"video": ("input.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert (config.web.storage_root / job_id / "input.mp4").exists()
    job_config = load_config(config.web.storage_root / job_id / "config.yaml")
    assert job_config.score.type == "gradient_diff"
    assert job_config.video.analysis_width == 64
    assert job_config.preprocess.fps == 2.0
    assert job_config.preprocess.height == 48
    assert job_config.render.fps == 12.0
    assert job_config.speed.type == "linear"
    assert job_config.speed.still_score == 0.5
    assert job_config.speed.motion_score == 8.5
    assert job_config.speed.min_speed == 1.5
    assert job_config.speed.max_speed == 20.0
    assert job_config.speed.sensitivity == 2.5
    assert job_config.speed.smoothing_window == 5

    payloads: list[dict[str, object]] = []
    with client.stream("GET", f"/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line.removeprefix("data: "))
            payloads.append(payload)
            if payload["status"] == "done":
                break

    assert [payload["status"] for payload in payloads] == [
        "queued",
        "running",
        "running",
        "done",
    ]
    assert payloads[-1]["timeline_url"] == f"/jobs/{job_id}/timeline.json"
    assert payloads[-1]["schedule_url"] == f"/jobs/{job_id}/schedule.json"
    assert payloads[-1]["replay_url"] == f"/jobs/{job_id}/replay.mp4"

    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.json()["status"] == "done"

    timeline_response = client.get(f"/jobs/{job_id}/timeline.json")
    schedule_response = client.get(f"/jobs/{job_id}/schedule.json")
    replay_response = client.get(f"/jobs/{job_id}/replay.mp4")
    assert timeline_response.status_code == 200
    assert schedule_response.status_code == 200
    assert replay_response.status_code == 200

    schedule = schedule_response.json()
    assert schedule["output_fps"] == 12.0
    assert schedule["total_source_seconds"] == 2.0
    assert schedule["total_replay_seconds"] == 1.5
    assert schedule["segments"][0]["output_frame_count"] == 6


def fake_processor(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> ProcessingResult:
    if progress_reporter is not None:
        progress_reporter(ProgressUpdate("timeline", 0.5, "Halfway"))

    write_timeline(
        config.output.json,
        Timeline(
            video=str(config.video.input),
            frame_interval_seconds=1.0,
            records=[
                TimelineRecord(time=0.0, score=0.0, replay_speed=2.0),
                TimelineRecord(time=1.0, score=10.0, replay_speed=1.0),
            ],
        ),
    )
    config.render.output_video.write_bytes(b"fake mp4")
    return ProcessingResult(
        timeline_path=config.output.json,
        replay_path=config.render.output_video,
    )


def web_test_config(tmp_path: Path) -> AppConfig:
    config_path = tmp_path / "config.yaml"
    write_runtime_config(
        path=config_path,
        video_path=tmp_path / "input.mp4",
        preprocessed_path=tmp_path / "preprocessed.mp4",
        timeline_path=tmp_path / "timeline.json",
        replay_path=tmp_path / "replay.mp4",
        web_storage_root=tmp_path / "web" / "jobs",
    )
    return load_config(config_path)
