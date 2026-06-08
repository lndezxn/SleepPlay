import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Condition
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

from sleepplay.config import AppConfig
from sleepplay.progress import ProgressReporter, ProgressUpdate
from sleepplay.schedule import build_replay_scale
from sleepplay.service import (
    ProcessingResult,
    ProcessingOverrides,
    config_for_job,
    process_replay,
    write_resolved_config,
)
from sleepplay.timeline import read_timeline
from sleepplay.video import video_fps

Processor = Callable[[AppConfig, ProgressReporter | None], ProcessingResult]


@dataclass
class JobState:
    job_id: str
    job_dir: Path
    input_path: Path
    config: AppConfig
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    events: list[dict[str, object]] = field(default_factory=list)
    condition: Condition = field(default_factory=Condition)

    @property
    def timeline_path(self) -> Path:
        return self.config.output.json

    @property
    def replay_path(self) -> Path:
        return self.config.render.output_video


class JobManager:
    def __init__(self, config: AppConfig, processor: Processor = process_replay) -> None:
        if config.web.max_workers <= 0:
            raise ValueError("web max_workers must be positive.")
        self.config = config
        self.processor = processor
        self.executor = ThreadPoolExecutor(max_workers=config.web.max_workers)
        self.jobs: dict[str, JobState] = {}

    def create_job(
        self,
        filename: str | None,
        overrides: ProcessingOverrides | None = None,
    ) -> JobState:
        job_id = uuid.uuid4().hex
        job_dir = self.config.web.storage_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "input.mp4").suffix or ".mp4"
        input_path = job_dir / f"input{suffix}"
        job_config = config_for_job(self.config, input_path, job_dir, overrides)
        write_resolved_config(job_dir / "config.yaml", job_config)

        job = JobState(
            job_id=job_id,
            job_dir=job_dir,
            input_path=input_path,
            config=job_config,
        )
        self.jobs[job_id] = job
        self.publish(job, "queued", "queued", 0.0, "Queued")
        return job

    def start_job(self, job: JobState) -> None:
        self.executor.submit(self.run_job, job)

    def run_job(self, job: JobState) -> None:
        self.publish(job, "running", "upload", 1.0, "Upload complete")

        def report(update: ProgressUpdate) -> None:
            self.publish(
                job,
                "running",
                update.stage,
                update.progress,
                update.message,
            )

        try:
            self.processor(job.config, report)
        except Exception as error:
            self.publish(job, "failed", "failed", 1.0, str(error))
            return

        self.publish(job, "done", "done", 1.0, "Replay ready")

    def publish(
        self,
        job: JobState,
        status: str,
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        with job.condition:
            job.status = status
            job.stage = stage
            job.progress = min(max(progress, 0.0), 1.0)
            job.message = message
            event = job_payload(job)
            job.events.append(event)
            job.condition.notify_all()

    def get_job(self, job_id: str) -> JobState:
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return self.jobs[job_id]

    def wait_for_event(self, job: JobState, event_index: int) -> dict[str, object]:
        with job.condition:
            while len(job.events) <= event_index:
                job.condition.wait()
            return job.events[event_index]


def create_app(
    config: AppConfig,
    processor: Processor = process_replay,
) -> FastAPI:
    app = FastAPI(title="SleepPlay")
    manager = JobManager(config, processor)
    app.state.job_manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(render_web_html(config))

    @app.post("/jobs")
    async def create_job(
        video: UploadFile = File(...),
        score_type: str | None = Form(None),
        analysis_width: int | None = Form(None),
        preprocess_fps: float | None = Form(None),
        preprocess_height: int | None = Form(None),
        render_source: str | None = Form(None),
        render_source_fps: float | None = Form(None),
        speed_type: str | None = Form(None),
        still_score: float | None = Form(None),
        motion_score: float | None = Form(None),
        min_speed: float | None = Form(None),
        max_speed: float | None = Form(None),
        sensitivity: float | None = Form(None),
        pooling_window: int | None = Form(None),
        smoothing_window: int | None = Form(None),
    ) -> dict[str, str]:
        job = manager.create_job(
            video.filename,
            ProcessingOverrides(
                analysis_width=analysis_width,
                preprocess_fps=preprocess_fps,
                preprocess_height=preprocess_height,
                render_source=render_source,
                render_source_fps=render_source_fps,
                score_type=score_type,
                speed_type=speed_type,
                still_score=still_score,
                motion_score=motion_score,
                min_speed=min_speed,
                max_speed=max_speed,
                sensitivity=sensitivity,
                pooling_window=pooling_window,
                smoothing_window=smoothing_window,
            ),
        )
        with job.input_path.open("wb") as file:
            while chunk := await video.read(1024 * 1024):
                file.write(chunk)
        manager.start_job(job)
        return {"job_id": job.job_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        return job_payload(manager.get_job(job_id))

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> EventSourceResponse:
        job = manager.get_job(job_id)

        async def event_stream() -> object:
            event_index = 0
            while True:
                event = await asyncio.to_thread(manager.wait_for_event, job, event_index)
                event_index += 1
                yield {
                    "event": str(event["status"]),
                    "data": json.dumps(event),
                }
                if event["status"] in {"done", "failed"}:
                    break

        return EventSourceResponse(event_stream())

    @app.get("/jobs/{job_id}/timeline.json")
    def timeline_json(job_id: str) -> FileResponse:
        job = manager.get_job(job_id)
        if not job.timeline_path.exists():
            raise HTTPException(status_code=404, detail="Timeline is not ready")
        return FileResponse(job.timeline_path, media_type="application/json")

    @app.get("/jobs/{job_id}/schedule.json")
    def replay_schedule(job_id: str) -> dict[str, object]:
        job = manager.get_job(job_id)
        if not job.timeline_path.exists():
            raise HTTPException(status_code=404, detail="Timeline is not ready")
        timeline = read_timeline(job.timeline_path)
        output_fps = video_fps(Path(timeline.render_video))
        segments = build_replay_scale(timeline, output_fps=output_fps)
        return {
            "output_fps": output_fps,
            "total_source_seconds": segments[-1].source_end,
            "total_replay_seconds": segments[-1].replay_end,
            "segments": [asdict(segment) for segment in segments],
        }

    @app.get("/jobs/{job_id}/replay.mp4")
    def replay_video(job_id: str) -> FileResponse:
        job = manager.get_job(job_id)
        if not job.replay_path.exists():
            raise HTTPException(status_code=404, detail="Replay is not ready")
        return FileResponse(job.replay_path, media_type="video/mp4")

    return app


def render_web_html(config: AppConfig) -> str:
    return WEB_HTML.replace("__DEFAULT_SETTINGS__", json.dumps(web_settings(config)))


def web_settings(config: AppConfig) -> dict[str, object]:
    return {
        "score_type": config.score.type,
        "analysis_width": config.video.analysis_width,
        "preprocess_fps": config.preprocess.fps,
        "preprocess_height": config.preprocess.height,
        "render_source": config.render.source,
        "render_source_fps": config.render.source_fps,
        "speed_type": config.speed.type,
        "still_score": config.speed.still_score,
        "motion_score": config.speed.motion_score,
        "min_speed": config.speed.min_speed,
        "max_speed": config.speed.max_speed,
        "sensitivity": config.speed.sensitivity,
        "pooling_window": config.speed.pooling_window,
        "smoothing_window": config.speed.smoothing_window,
    }


def job_payload(job: JobState) -> dict[str, object]:
    payload: dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
    }
    if job.timeline_path.exists():
        payload["timeline_url"] = f"/jobs/{job.job_id}/timeline.json"
        payload["schedule_url"] = f"/jobs/{job.job_id}/schedule.json"
    if job.replay_path.exists():
        payload["replay_url"] = f"/jobs/{job.job_id}/replay.mp4"
    return payload


WEB_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>SleepPlay</title>
    <script>
      const storedTheme = window.localStorage.getItem("sleepplay.theme");
      const initialTheme = storedTheme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      document.documentElement.dataset.theme = initialTheme;
    </script>
    <style>
      :root {
        color-scheme: light;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        --page: #fff7f7;
        --page-gradient-start: #fff1f2;
        --page-gradient-end: #fffafa;
        --panel: #fffdfd;
        --panel-soft: #fff4f4;
        --field: #fffafa;
        --field-hover: #fff4f4;
        --text: #2c1e22;
        --muted: #735c62;
        --line: #efd0d5;
        --line-strong: #ddaeb7;
        --red: #c8324a;
        --red-dark: #9f2639;
        --red-soft: #fde6e9;
        --red-track: #f5d6dc;
        --button-text: #ffffff;
        --focus-ring: rgba(200, 50, 74, 0.22);
        --stage-fill-start: #e47b8b;
        --score: #c3465d;
        --cursor: #351d24;
        --preview: #b98b94;
        --shadow: rgba(99, 32, 45, 0.12);
        --pill-shadow: rgba(145, 43, 61, 0.08);
        --video-bg: #1b1114;
        --video-shadow: rgba(62, 16, 27, 0.12);
        --control-bg: #fffafa;
        --control-shadow: rgba(104, 28, 42, 0.1);
        --control-active-shadow: rgba(104, 28, 42, 0.12);
        --timeline-bg-start: #fffafa;
        --timeline-bg-end: #fff5f6;
        --timeline-hover-start: #fff8f9;
        --timeline-time-bg: rgba(255, 248, 249, 0.9);
        --timeline-time-border: rgba(239, 208, 213, 0.78);
        --timeline-inset: rgba(255, 255, 255, 0.82);
        background: var(--page);
        color: var(--text);
      }
      :root[data-theme="dark"] {
        color-scheme: dark;
        --page: #170b0e;
        --page-gradient-start: #241016;
        --page-gradient-end: #13090d;
        --panel: #211116;
        --panel-soft: #2a151c;
        --field: #2a151c;
        --field-hover: #321923;
        --text: #fff2f4;
        --muted: #caa8b0;
        --line: #4b2b35;
        --line-strong: #7a4352;
        --red: #ef5b73;
        --red-dark: #ff9aaa;
        --red-soft: #3c1a23;
        --red-track: #3a2028;
        --button-text: #1a0c10;
        --focus-ring: rgba(239, 91, 115, 0.28);
        --stage-fill-start: #ad314a;
        --score: #ff6f87;
        --cursor: #fff7f8;
        --preview: #d5a8b1;
        --shadow: rgba(0, 0, 0, 0.32);
        --pill-shadow: rgba(0, 0, 0, 0.22);
        --video-bg: #090506;
        --video-shadow: rgba(0, 0, 0, 0.36);
        --control-bg: #2a151c;
        --control-shadow: rgba(0, 0, 0, 0.28);
        --control-active-shadow: rgba(0, 0, 0, 0.24);
        --timeline-bg-start: #29141b;
        --timeline-bg-end: #1d0e13;
        --timeline-hover-start: #321923;
        --timeline-time-bg: rgba(33, 17, 22, 0.9);
        --timeline-time-border: rgba(122, 67, 82, 0.82);
        --timeline-inset: rgba(255, 255, 255, 0.07);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        background:
          linear-gradient(180deg, var(--page-gradient-start) 0%, var(--page) 42%, var(--page-gradient-end) 100%);
      }
      main {
        width: min(1380px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 24px 0 40px;
      }
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
      }
      .header-actions {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      h1 {
        margin: 0;
        font-size: 24px;
        line-height: 1.1;
        font-weight: 680;
        color: var(--red-dark);
      }
      .status-pill {
        min-width: 128px;
        text-align: center;
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        color: var(--muted);
        box-shadow: 0 3px 10px var(--pill-shadow);
      }
      .theme-button {
        display: inline-grid;
        place-items: center;
        width: 42px;
        min-width: 42px;
        height: 42px;
        min-height: 42px;
        padding: 0;
        border: 1px solid var(--line);
        border-radius: 50%;
        background: var(--panel);
        color: var(--red-dark);
        box-shadow: 0 3px 10px var(--pill-shadow);
      }
      .theme-button:hover {
        background: var(--panel-soft);
        border-color: var(--line-strong);
        color: var(--red);
      }
      .theme-icon {
        width: 19px;
        height: 19px;
        stroke: currentColor;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 14px 34px var(--shadow);
      }
      .upload-panel {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        align-items: center;
        margin-bottom: 18px;
      }
      .workspace {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 328px;
        gap: 18px;
        align-items: start;
      }
      .main-column {
        min-width: 0;
      }
      input[type="file"] {
        width: 100%;
        min-height: 42px;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 8px;
        background: var(--field);
        color: var(--text);
      }
      input[type="file"]:hover {
        border-color: var(--line-strong);
        background: var(--field-hover);
      }
      button {
        min-height: 42px;
        border: 0;
        border-radius: 6px;
        background: var(--red);
        color: var(--button-text);
        padding: 0 18px;
        font-weight: 650;
        cursor: pointer;
        transition: background 140ms ease, border-color 140ms ease, color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
      }
      button:hover {
        background: var(--red-dark);
      }
      button:active {
        transform: translateY(1px);
      }
      button:disabled {
        cursor: not-allowed;
        opacity: 0.56;
      }
      button:focus-visible {
        outline: 3px solid var(--focus-ring);
        outline-offset: 2px;
      }
      .stage-progress {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }
      .stage-step {
        display: grid;
        gap: 6px;
        min-width: 0;
      }
      .stage-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
      }
      .stage-step.active .stage-label {
        color: var(--red-dark);
      }
      .stage-shell {
        height: 7px;
        overflow: hidden;
        border-radius: 999px;
        background: var(--red-track);
      }
      .stage-fill {
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, var(--stage-fill-start), var(--red));
        transition: width 160ms ease;
      }
      .message {
        min-height: 20px;
        margin-top: 10px;
        color: var(--muted);
        font-size: 14px;
      }
      .error {
        color: var(--red-dark);
      }
      .settings-sidebar {
        position: sticky;
        top: 18px;
        display: grid;
        gap: 14px;
      }
      .settings-title {
        margin: 0;
        color: var(--red-dark);
        font-size: 16px;
        line-height: 1.2;
        font-weight: 700;
      }
      .settings-group {
        display: grid;
        gap: 10px;
      }
      .settings-group-title {
        margin: 0;
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .settings-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .settings-field {
        display: grid;
        gap: 5px;
        min-width: 0;
      }
      .settings-field.full {
        grid-column: 1 / -1;
      }
      .settings-field label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
      }
      .settings-field input,
      .settings-field select {
        width: 100%;
        min-height: 36px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--field);
        color: var(--text);
        padding: 0 9px;
        font: inherit;
      }
      .settings-field input:focus,
      .settings-field select:focus {
        border-color: var(--line-strong);
        outline: 3px solid var(--focus-ring);
      }
      .result {
        display: none;
        gap: 16px;
      }
      .result.ready {
        display: grid;
      }
      video {
        display: block;
        width: 100%;
        height: min(58vh, 66vw);
        max-height: 58vh;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        object-position: center center;
        background: var(--video-bg);
        border-radius: 8px;
        box-shadow: 0 1px 0 var(--line), 0 16px 30px var(--video-shadow);
      }
      .player-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        padding: 8px 0 4px;
      }
      .control-button {
        display: inline-grid;
        place-items: center;
        width: 42px;
        min-width: 42px;
        height: 42px;
        min-height: 42px;
        padding: 0;
        border: 1px solid var(--line);
        border-radius: 50%;
        background: var(--control-bg);
        color: var(--red-dark);
        box-shadow: 0 5px 14px var(--control-shadow);
      }
      .control-button:hover {
        background: var(--red-soft);
        border-color: var(--line-strong);
        color: var(--red);
        transform: translateY(-1px);
      }
      .control-button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 7px var(--control-active-shadow);
      }
      .control-button.primary {
        width: 50px;
        min-width: 50px;
        height: 50px;
        min-height: 50px;
        background: var(--red);
        border-color: var(--red);
        color: var(--button-text);
        box-shadow: 0 9px 22px rgba(200, 50, 74, 0.24);
      }
      .control-button.primary:hover {
        background: var(--red-dark);
        border-color: var(--red-dark);
        color: var(--button-text);
      }
      .control-icon {
        width: 20px;
        height: 20px;
        stroke: currentColor;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
      }
      .control-button.primary .control-icon {
        width: 24px;
        height: 24px;
      }
      .control-spacer {
        width: 16px;
      }
      .timeline-stack {
        display: grid;
        gap: 16px;
        padding-top: 2px;
      }
      .timeline-row {
        display: grid;
        grid-template-columns: 96px 1fr;
        gap: 12px;
        align-items: center;
      }
      .timeline-label {
        color: var(--red-dark);
        font-size: 13px;
        font-weight: 650;
      }
      .timeline-track {
        position: relative;
        min-width: 0;
      }
      .timeline-time {
        position: absolute;
        top: 6px;
        right: 9px;
        padding: 2px 6px;
        border-radius: 5px;
        background: var(--timeline-time-bg);
        color: var(--red-dark);
        font-size: 12px;
        font-variant-numeric: tabular-nums;
        pointer-events: none;
        border: 1px solid var(--timeline-time-border);
      }
      .timeline-preview {
        position: absolute;
        left: 0;
        bottom: 6px;
        transform: translateX(-50%);
        display: none;
        padding: 2px 6px;
        border-radius: 5px;
        background: rgba(130, 39, 57, 0.86);
        color: #ffffff;
        font-size: 12px;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        pointer-events: none;
      }
      .timeline-preview.visible {
        display: block;
      }
      svg.timeline {
        width: 100%;
        height: 64px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: linear-gradient(180deg, var(--timeline-bg-start) 0%, var(--timeline-bg-end) 100%);
        cursor: pointer;
        display: block;
        overflow: visible;
        box-shadow: inset 0 1px 0 var(--timeline-inset);
      }
      svg.timeline:hover {
        border-color: var(--line-strong);
        background: linear-gradient(180deg, var(--timeline-hover-start) 0%, var(--red-soft) 100%);
      }
      .result:fullscreen {
        position: relative;
        display: block;
        width: 100vw;
        height: 100vh;
        max-width: none;
        border: 0;
        border-radius: 0;
        padding: 0;
        background: #170b0e;
        box-shadow: none;
        overflow: hidden;
      }
      .result:fullscreen video {
        width: 100vw;
        height: 100vh;
        max-height: none;
        border-radius: 0;
        box-shadow: none;
      }
      .result:fullscreen .player-controls,
      .result:fullscreen .timeline-stack {
        position: absolute;
        left: 18px;
        right: 18px;
        z-index: 5;
        opacity: 0;
        pointer-events: none;
        transform: translateY(10px);
        transition: opacity 180ms ease, transform 180ms ease;
      }
      .result:fullscreen.fullscreen-ui-visible .player-controls,
      .result:fullscreen.fullscreen-ui-visible .timeline-stack {
        opacity: 1;
        pointer-events: auto;
        transform: translateY(0);
      }
      .result:fullscreen .player-controls {
        bottom: 136px;
        display: flex;
        width: fit-content;
        min-width: min(360px, calc(100vw - 36px));
        justify-content: center;
        margin: 0 auto;
        padding: 10px 12px;
        border: 1px solid rgba(239, 208, 213, 0.28);
        border-radius: 999px;
        background: rgba(43, 14, 20, 0.56);
        backdrop-filter: blur(12px);
        box-shadow: 0 14px 42px rgba(23, 11, 14, 0.28);
      }
      .result:fullscreen .timeline-stack {
        bottom: 18px;
        gap: 8px;
        padding: 10px 12px;
        border: 1px solid rgba(239, 208, 213, 0.26);
        border-radius: 8px;
        background: rgba(43, 14, 20, 0.52);
        backdrop-filter: blur(12px);
        box-shadow: 0 14px 42px rgba(23, 11, 14, 0.28);
      }
      .result:fullscreen .timeline-row {
        grid-template-columns: 72px 1fr;
        gap: 8px;
      }
      .result:fullscreen .timeline-label {
        color: rgba(255, 241, 243, 0.92);
      }
      .result:fullscreen svg.timeline {
        height: 30px;
        border-color: rgba(255, 225, 230, 0.28);
        background: rgba(255, 244, 246, 0.14);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
      }
      .result:fullscreen svg.timeline:hover {
        border-color: rgba(255, 225, 230, 0.48);
        background: rgba(255, 244, 246, 0.2);
      }
      .result:fullscreen .timeline-time {
        top: 4px;
        background: rgba(43, 14, 20, 0.56);
        color: rgba(255, 247, 248, 0.96);
        border-color: rgba(255, 225, 230, 0.24);
      }
      .result:fullscreen .timeline-preview {
        bottom: 4px;
        background: rgba(200, 50, 74, 0.82);
      }
      @media (max-width: 680px) {
        main {
          width: min(100vw - 20px, 1180px);
          padding-top: 14px;
        }
        header,
        .workspace,
        .upload-panel,
        .player-controls,
        .timeline-row {
          grid-template-columns: 1fr;
          display: grid;
        }
        .settings-sidebar {
          position: static;
        }
        .stage-progress {
          grid-template-columns: 1fr 1fr;
        }
        .timeline-row {
          gap: 6px;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>SleepPlay</h1>
        <div class="header-actions">
          <button id="theme-button" class="theme-button" type="button" aria-label="Switch to dark mode" title="Switch to dark mode">
            <svg id="theme-icon" class="theme-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"></path>
            </svg>
          </button>
          <div id="status-pill" class="status-pill">Idle</div>
        </div>
      </header>

      <div class="workspace">
        <div class="main-column">
          <section class="panel upload-panel">
            <form id="upload-form" style="display: contents;">
              <input id="video-input" name="video" type="file" accept="video/*" required>
              <button id="upload-button" type="submit">Process</button>
            </form>
            <div style="grid-column: 1 / -1;">
              <div class="stage-progress" aria-label="Processing progress">
                <div id="stage-upload" class="stage-step" data-stage="upload">
                  <div class="stage-label">Upload</div>
                  <div class="stage-shell"><div class="stage-fill"></div></div>
                </div>
                <div id="stage-preprocess" class="stage-step" data-stage="preprocess">
                  <div class="stage-label">Preprocess</div>
                  <div class="stage-shell"><div class="stage-fill"></div></div>
                </div>
                <div id="stage-timeline" class="stage-step" data-stage="timeline">
                  <div class="stage-label">Analyze</div>
                  <div class="stage-shell"><div class="stage-fill"></div></div>
                </div>
                <div id="stage-render" class="stage-step" data-stage="render">
                  <div class="stage-label">Render</div>
                  <div class="stage-shell"><div class="stage-fill"></div></div>
                </div>
              </div>
              <div id="message" class="message"></div>
            </div>
          </section>

          <section id="result" class="panel result">
            <video id="replay-video" preload="metadata"></video>
            <div class="player-controls">
              <button id="reset-button" class="control-button" type="button" aria-label="Reset" title="Reset">
                <svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M3 12a9 9 0 1 0 3-6.7"></path>
                  <path d="M3 4v6h6"></path>
                </svg>
              </button>
              <button id="back-button" class="control-button" type="button" aria-label="Back 10 seconds" title="Back 10 seconds">
                <svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M11 7 5 12l6 5V7Z"></path>
                  <path d="M19 7 13 12l6 5V7Z"></path>
                </svg>
              </button>
              <button id="play-button" class="control-button primary" type="button" aria-label="Play" title="Play">
                <svg id="play-icon" class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8 5v14l11-7-11-7Z"></path>
                </svg>
              </button>
              <button id="forward-button" class="control-button" type="button" aria-label="Forward 10 seconds" title="Forward 10 seconds">
                <svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="m5 7 6 5-6 5V7Z"></path>
                  <path d="m13 7 6 5-6 5V7Z"></path>
                </svg>
              </button>
              <span class="control-spacer" aria-hidden="true"></span>
              <button id="fullscreen-button" class="control-button" type="button" aria-label="Fullscreen" title="Fullscreen">
                <svg id="fullscreen-icon" class="control-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8 3H3v5"></path>
                  <path d="M16 3h5v5"></path>
                  <path d="M21 16v5h-5"></path>
                  <path d="M8 21H3v-5"></path>
                </svg>
              </button>
            </div>
            <div class="timeline-stack">
              <div class="timeline-row">
                <div class="timeline-label">Original</div>
                <div class="timeline-track">
                  <svg id="source-timeline" class="timeline" role="img" aria-label="Original timeline"></svg>
                  <div id="source-time" class="timeline-time">0:00 / 0:00</div>
                  <div id="source-preview" class="timeline-preview">0:00</div>
                </div>
              </div>
              <div class="timeline-row">
                <div class="timeline-label">Replay</div>
                <div class="timeline-track">
                  <svg id="replay-timeline" class="timeline" role="img" aria-label="Replay timeline"></svg>
                  <div id="replay-time" class="timeline-time">0:00 / 0:00</div>
                  <div id="replay-preview" class="timeline-preview">0:00</div>
                </div>
              </div>
            </div>
          </section>
        </div>

        <aside class="panel settings-sidebar" aria-label="Processing settings">
          <h2 class="settings-title">Processing Settings</h2>
          <section class="settings-group">
            <h3 class="settings-group-title">Score</h3>
            <div class="settings-grid">
              <div class="settings-field full">
                <label for="score-type">Algorithm</label>
                <select id="score-type" name="score_type" form="upload-form">
                  <option value="frame_diff">Frame diff</option>
                  <option value="gradient_diff">Gradient diff</option>
                </select>
              </div>
              <div class="settings-field">
                <label for="analysis-width">Analysis width</label>
                <input id="analysis-width" name="analysis_width" form="upload-form" type="number" min="16" step="1">
              </div>
              <div class="settings-field">
                <label for="preprocess-height">Preprocess height</label>
                <input id="preprocess-height" name="preprocess_height" form="upload-form" type="number" min="16" step="1">
              </div>
              <div class="settings-field">
                <label for="preprocess-fps">Preprocess FPS</label>
                <input id="preprocess-fps" name="preprocess_fps" form="upload-form" type="number" min="0.1" step="0.1">
              </div>
              <div class="settings-field full">
                <label for="render-source">Render source</label>
                <select id="render-source" name="render_source" form="upload-form">
                  <option value="preprocessed">Preprocessed</option>
                  <option value="original">Original</option>
                </select>
              </div>
              <div class="settings-field">
                <label for="render-source-fps">Source FPS</label>
                <input id="render-source-fps" name="render_source_fps" form="upload-form" type="number" min="0.1" step="0.1">
              </div>
            </div>
          </section>
          <section class="settings-group">
            <h3 class="settings-group-title">Speed</h3>
            <div class="settings-grid">
              <div class="settings-field full">
                <label for="speed-type">Mapping</label>
                <select id="speed-type" name="speed_type" form="upload-form">
                  <option value="sensitive">Sensitive</option>
                  <option value="linear">Linear</option>
                </select>
              </div>
              <div class="settings-field">
                <label for="still-score">Still score</label>
                <input id="still-score" name="still_score" form="upload-form" type="number" step="0.1">
              </div>
              <div class="settings-field">
                <label for="motion-score">Motion score</label>
                <input id="motion-score" name="motion_score" form="upload-form" type="number" step="0.1">
              </div>
              <div class="settings-field">
                <label for="min-speed">Min speed</label>
                <input id="min-speed" name="min_speed" form="upload-form" type="number" min="0.1" step="0.1">
              </div>
              <div class="settings-field">
                <label for="max-speed">Max speed</label>
                <input id="max-speed" name="max_speed" form="upload-form" type="number" min="0.1" step="0.1">
              </div>
              <div class="settings-field">
                <label for="sensitivity">Sensitivity</label>
                <input id="sensitivity" name="sensitivity" form="upload-form" type="number" min="0.1" step="0.1">
              </div>
              <div class="settings-field">
                <label for="pooling-window">Pooling</label>
                <input id="pooling-window" name="pooling_window" form="upload-form" type="number" min="1" step="1">
              </div>
              <div class="settings-field">
                <label for="smoothing-window">Smoothing</label>
                <input id="smoothing-window" name="smoothing_window" form="upload-form" type="number" min="1" step="1">
              </div>
            </div>
          </section>
        </aside>
      </div>
    </main>

    <script>
      const form = document.querySelector("#upload-form");
      const input = document.querySelector("#video-input");
      const button = document.querySelector("#upload-button");
      const themeButton = document.querySelector("#theme-button");
      const themeIcon = document.querySelector("#theme-icon");
      const statusPill = document.querySelector("#status-pill");
      const message = document.querySelector("#message");
      const result = document.querySelector("#result");
      const video = document.querySelector("#replay-video");
      const resetButton = document.querySelector("#reset-button");
      const backButton = document.querySelector("#back-button");
      const playButton = document.querySelector("#play-button");
      const playIcon = document.querySelector("#play-icon");
      const forwardButton = document.querySelector("#forward-button");
      const fullscreenButton = document.querySelector("#fullscreen-button");
      const fullscreenIcon = document.querySelector("#fullscreen-icon");
      const sourceTimeText = document.querySelector("#source-time");
      const replayTimeText = document.querySelector("#replay-time");
      const sourcePreview = document.querySelector("#source-preview");
      const replayPreview = document.querySelector("#replay-preview");
      const sourceTimeline = document.querySelector("#source-timeline");
      const replayTimeline = document.querySelector("#replay-timeline");

      const defaultSettings = __DEFAULT_SETTINGS__;
      const themeStorageKey = "sleepplay.theme";
      const themePreference = window.matchMedia("(prefers-color-scheme: dark)");
      const progressStages = ["upload", "preprocess", "timeline", "render"];
      const stageElements = Object.fromEntries(
        progressStages.map((stage) => [stage, document.querySelector(`[data-stage="${stage}"]`)])
      );
      const buttonSeekSeconds = 10;
      const keyboardSeekSeconds = 1;
      let timeline = null;
      let segments = [];
      let totalSource = 1;
      let totalReplay = 1;
      let outputFps = 30;
      let hoverPreview = null;
      let fullscreenHideTimer = null;

      hydrateSettings();
      syncThemeButton(currentTheme());

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!input.files.length) return;
        resetUi();
        const formData = new FormData(form);
        const response = await fetch("/jobs", { method: "POST", body: formData });
        if (!response.ok) {
          showFailure("Upload failed");
          return;
        }
        const payload = await response.json();
        connectEvents(payload.job_id);
      });

      themeButton.addEventListener("click", () => {
        setTheme(currentTheme() === "dark" ? "light" : "dark", true);
      });
      themePreference.addEventListener("change", (event) => {
        if (window.localStorage.getItem(themeStorageKey) !== null) return;
        setTheme(event.matches ? "dark" : "light", false);
      });
      playButton.addEventListener("click", togglePlayback);
      resetButton.addEventListener("click", () => seekReplay(0));
      backButton.addEventListener("click", () => {
        seekReplay((video.currentTime || 0) - buttonSeekSeconds);
      });
      forwardButton.addEventListener("click", () => {
        seekReplay((video.currentTime || 0) + buttonSeekSeconds);
      });
      fullscreenButton.addEventListener("click", toggleFullscreen);
      document.addEventListener("fullscreenchange", updateFullscreenButton);
      result.addEventListener("pointermove", showFullscreenChrome);
      result.addEventListener("pointerdown", showFullscreenChrome);
      result.addEventListener("pointerleave", hideFullscreenChrome);
      document.addEventListener("keydown", (event) => {
        if (!segments.length) return;
        if (event.target instanceof HTMLInputElement) return;
        if (event.code === "Space") {
          event.preventDefault();
          togglePlayback();
          return;
        }
        if (event.code === "ArrowLeft") {
          event.preventDefault();
          seekReplay((video.currentTime || 0) - keyboardSeekSeconds);
          return;
        }
        if (event.code === "ArrowRight") {
          event.preventDefault();
          seekReplay((video.currentTime || 0) + keyboardSeekSeconds);
        }
      });

      video.addEventListener("play", () => {
        setPlayButtonState(true);
        drawLoop();
      });
      video.addEventListener("pause", () => {
        setPlayButtonState(false);
        drawAll();
      });
      video.addEventListener("ended", () => {
        setPlayButtonState(false);
        drawAll();
      });
      video.addEventListener("timeupdate", drawAll);
      window.addEventListener("resize", drawAll);

      attachTimelineDrag(sourceTimeline, "source");
      attachTimelineDrag(replayTimeline, "replay");

      function hydrateSettings() {
        Object.entries(defaultSettings).forEach(([name, value]) => {
          const field = document.querySelector(`[name="${name}"]`);
          if (field !== null) field.value = String(value);
        });
      }

      function currentTheme() {
        return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
      }

      function setTheme(theme, shouldPersist) {
        document.documentElement.dataset.theme = theme;
        if (shouldPersist) window.localStorage.setItem(themeStorageKey, theme);
        syncThemeButton(theme);
        drawAll();
      }

      function syncThemeButton(theme) {
        const isDark = theme === "dark";
        themeButton.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
        themeButton.setAttribute("title", isDark ? "Switch to light mode" : "Switch to dark mode");
        themeIcon.innerHTML = isDark
          ? '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.9 4.9 1.4 1.4"></path><path d="m17.7 17.7 1.4 1.4"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.3 17.7-1.4 1.4"></path><path d="m19.1 4.9-1.4 1.4"></path>'
          : '<path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"></path>';
      }

      function resetUi() {
        button.disabled = true;
        statusPill.textContent = "Queued";
        message.classList.remove("error");
        message.textContent = "Uploading";
        resetStageProgress();
        result.classList.remove("ready");
        video.removeAttribute("src");
        video.currentTime = 0;
        sourceTimeText.textContent = "0:00 / 0:00";
        replayTimeText.textContent = "0:00 / 0:00";
        clearTimelinePreview();
        setPlayButtonState(false);
        timeline = null;
        segments = [];
      }

      function connectEvents(jobId) {
        const events = new EventSource(`/jobs/${jobId}/events`);
        const handle = (event) => {
          const payload = JSON.parse(event.data);
          updateStatus(payload);
          if (payload.status === "done") {
            events.close();
            loadResult(payload);
          }
          if (payload.status === "failed") {
            events.close();
            showFailure(payload.message || "Processing failed");
          }
        };
        ["queued", "running", "done", "failed"].forEach((name) => {
          events.addEventListener(name, handle);
        });
        events.onerror = () => showFailure("Connection lost");
      }

      function updateStatus(payload) {
        statusPill.textContent = payload.stage || payload.status;
        message.textContent = payload.message || "";
        updateStageProgress(payload.stage, payload.progress || 0, payload.status);
      }

      function resetStageProgress() {
        progressStages.forEach((stage) => {
          setStageProgress(stage, 0);
          stageElements[stage].classList.remove("active");
        });
      }

      function updateStageProgress(stage, progress, status) {
        if (status === "done") {
          progressStages.forEach((progressStage) => setStageProgress(progressStage, 1));
          progressStages.forEach((progressStage) => {
            stageElements[progressStage].classList.remove("active");
          });
          return;
        }
        if (!progressStages.includes(stage)) return;

        const activeIndex = progressStages.indexOf(stage);
        progressStages.forEach((progressStage, index) => {
          stageElements[progressStage].classList.toggle("active", progressStage === stage);
          if (index < activeIndex) {
            setStageProgress(progressStage, 1);
          }
        });
        setStageProgress(stage, progress);
      }

      function setStageProgress(stage, progress) {
        const fill = stageElements[stage].querySelector(".stage-fill");
        fill.style.width = `${Math.round(clamp(progress, 0, 1) * 100)}%`;
      }

      async function loadResult(payload) {
        button.disabled = false;
        video.src = `${payload.replay_url}?v=${Date.now()}`;
        const response = await fetch(`${payload.schedule_url}?v=${Date.now()}`);
        timeline = await response.json();
        outputFps = timeline.output_fps;
        segments = buildSegments(timeline);
        totalSource = timeline.total_source_seconds;
        totalReplay = timeline.total_replay_seconds;
        result.classList.add("ready");
        video.load();
        drawAll();
      }

      function showFailure(text) {
        button.disabled = false;
        statusPill.textContent = "Failed";
        message.classList.add("error");
        message.textContent = text;
      }

      function buildSegments(timelineData) {
        return timelineData.segments.map((segment) => ({
          sourceStart: segment.source_start,
          sourceEnd: segment.source_end,
          replayStart: segment.replay_start,
          replayEnd: segment.replay_end,
          score: segment.score,
          speed: segment.replay_speed,
          outputFrameCount: segment.output_frame_count,
          outputFps: segment.output_fps
        }));
      }

      function sourceToReplay(sourceTime) {
        const segment = segmentForSource(sourceTime);
        const frameIndex = Math.round(
          ((sourceTime - segment.sourceStart) / segment.speed) * segment.outputFps
        );
        const clampedFrameIndex = clamp(frameIndex, 0, segment.outputFrameCount - 1);
        return segment.replayStart + clampedFrameIndex / segment.outputFps;
      }

      function replayToSource(replayTime) {
        const segment = segmentForReplay(replayTime);
        const frameIndex = Math.floor((replayTime - segment.replayStart) * segment.outputFps);
        const clampedFrameIndex = clamp(frameIndex, 0, segment.outputFrameCount - 1);
        return Math.min(
          segment.sourceStart + clampedFrameIndex / segment.outputFps * segment.speed,
          segment.sourceEnd
        );
      }

      function segmentForSource(sourceTime) {
        const time = clamp(sourceTime, 0, totalSource);
        return segments.find((segment, index) => {
          const isLast = index === segments.length - 1;
          return time >= segment.sourceStart && (time < segment.sourceEnd || (isLast && time <= segment.sourceEnd));
        }) || segments[segments.length - 1];
      }

      function segmentForReplay(replayTime) {
        const time = clamp(replayTime, 0, totalReplay);
        return segments.find((segment, index) => {
          const isLast = index === segments.length - 1;
          return time >= segment.replayStart && (time < segment.replayEnd || (isLast && time <= segment.replayEnd));
        }) || segments[segments.length - 1];
      }

      function attachTimelineDrag(svg, scale) {
        const preview = (event) => {
          if (!segments.length) return;
          setTimelinePreview(timePairFromPointer(svg, scale, event));
        };
        const seek = (event) => {
          if (!segments.length) return;
          const times = timePairFromPointer(svg, scale, event);
          setTimelinePreview(times);
          seekReplay(times.replayTime);
        };
        svg.addEventListener("pointerenter", preview);
        svg.addEventListener("pointermove", preview);
        svg.addEventListener("pointerleave", clearTimelinePreview);
        svg.addEventListener("pointerdown", (event) => {
          svg.setPointerCapture(event.pointerId);
          seek(event);
        });
        svg.addEventListener("pointermove", (event) => {
          if (event.buttons === 1) seek(event);
        });
      }

      function drawLoop() {
        drawAll();
        if (!video.paused) requestAnimationFrame(drawLoop);
      }

      function drawAll() {
        if (!segments.length) return;
        drawTimeline(sourceTimeline, "source");
        drawTimeline(replayTimeline, "replay");
        const replayTime = clamp(video.currentTime || 0, 0, totalReplay);
        const sourceTime = replayToSource(replayTime);
        sourceTimeText.textContent = `${formatTime(sourceTime)} / ${formatTime(totalSource)}`;
        replayTimeText.textContent = `${formatTime(replayTime)} / ${formatTime(totalReplay)}`;
      }

      function drawTimeline(svg, scale) {
        const duration = scale === "source" ? totalSource : totalReplay;
        svg.setAttribute("viewBox", `0 0 ${duration} 1`);
        svg.setAttribute("preserveAspectRatio", "none");
        svg.innerHTML = "";
        svg.appendChild(pathElement(buildScorePath(scale), "var(--score)"));
        if (hoverPreview !== null) {
          svg.appendChild(previewCursorElement(scale));
        }
        svg.appendChild(cursorElement(scale));
      }

      function buildScorePath(scale) {
        const values = segments.map((segment) => segment.score);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;
        const commands = [];
        segments.forEach((segment) => {
          const start = scale === "source" ? segment.sourceStart : segment.replayStart;
          const end = scale === "source" ? segment.sourceEnd : segment.replayEnd;
          const valueRatio = valueRange === 0 ? 0.5 : (segment.score - minValue) / valueRange;
          const y = 1 - valueRatio;
          if (!commands.length) commands.push(`M ${start} ${y}`);
          else commands.push(`L ${start} ${y}`);
          commands.push(`L ${end} ${y}`);
        });
        return commands.join(" ");
      }

      function pathElement(pathData, color) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathData);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "2");
        path.setAttribute("vector-effect", "non-scaling-stroke");
        path.setAttribute("stroke-linecap", "butt");
        path.setAttribute("stroke-linejoin", "round");
        return path;
      }

      function cursorElement(scale) {
        const replayTime = clamp(video.currentTime || 0, 0, totalReplay);
        const sourceTime = replayToSource(replayTime);
        const x = scale === "source" ? sourceTime : replayTime;
        return verticalLineElement(x, "var(--cursor)", "2");
      }

      function previewCursorElement(scale) {
        const x = scale === "source" ? hoverPreview.sourceTime : hoverPreview.replayTime;
        return verticalLineElement(x, "var(--preview)", "2");
      }

      function verticalLineElement(x, color, width) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x);
        line.setAttribute("x2", x);
        line.setAttribute("y1", "0");
        line.setAttribute("y2", "1");
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", width);
        line.setAttribute("vector-effect", "non-scaling-stroke");
        return line;
      }

      function seekReplay(replayTime) {
        if (!segments.length) return;
        video.currentTime = clamp(replayTime, 0, totalReplay);
        drawAll();
      }

      async function togglePlayback() {
        if (!segments.length) return;
        if (video.paused) {
          await video.play();
        } else {
          video.pause();
        }
      }

      function setPlayButtonState(isPlaying) {
        playButton.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
        playButton.setAttribute("title", isPlaying ? "Pause" : "Play");
        playIcon.innerHTML = isPlaying
          ? '<path d="M8 5v14"></path><path d="M16 5v14"></path>'
          : '<path d="M8 5v14l11-7-11-7Z"></path>';
      }

      async function toggleFullscreen() {
        if (document.fullscreenElement === result) {
          await document.exitFullscreen();
          return;
        }
        await result.requestFullscreen();
      }

      function updateFullscreenButton() {
        const isFullscreen = document.fullscreenElement === result;
        fullscreenButton.setAttribute("aria-label", isFullscreen ? "Exit fullscreen" : "Fullscreen");
        fullscreenButton.setAttribute("title", isFullscreen ? "Exit fullscreen" : "Fullscreen");
        fullscreenIcon.innerHTML = isFullscreen
          ? '<path d="M9 3v6H3"></path><path d="M15 3v6h6"></path><path d="M21 15h-6v6"></path><path d="M3 15h6v6"></path>'
          : '<path d="M8 3H3v5"></path><path d="M16 3h5v5"></path><path d="M21 16v5h-5"></path><path d="M8 21H3v-5"></path>';
        if (isFullscreen) {
          showFullscreenChrome();
        } else {
          window.clearTimeout(fullscreenHideTimer);
          result.classList.remove("fullscreen-ui-visible");
        }
      }

      function showFullscreenChrome() {
        if (document.fullscreenElement !== result) return;
        result.classList.add("fullscreen-ui-visible");
        window.clearTimeout(fullscreenHideTimer);
        fullscreenHideTimer = window.setTimeout(hideFullscreenChrome, 1600);
      }

      function hideFullscreenChrome() {
        if (document.fullscreenElement !== result) return;
        window.clearTimeout(fullscreenHideTimer);
        result.classList.remove("fullscreen-ui-visible");
      }

      function timePairFromPointer(svg, scale, event) {
        const rect = svg.getBoundingClientRect();
        const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        if (scale === "source") {
          const sourceTime = ratio * totalSource;
          return {
            sourceTime,
            replayTime: sourceToReplay(sourceTime)
          };
        }

        const replayTime = ratio * totalReplay;
        return {
          sourceTime: replayToSource(replayTime),
          replayTime
        };
      }

      function setTimelinePreview(times) {
        hoverPreview = {
          sourceTime: clamp(times.sourceTime, 0, totalSource),
          replayTime: clamp(times.replayTime, 0, totalReplay)
        };
        const score = scoreForSourceTime(hoverPreview.sourceTime);
        updatePreviewLabel(sourcePreview, hoverPreview.sourceTime, totalSource, score);
        updatePreviewLabel(replayPreview, hoverPreview.replayTime, totalReplay, score);
        drawAll();
      }

      function updatePreviewLabel(element, time, duration, score) {
        const ratio = duration === 0 ? 0 : clamp(time / duration, 0, 1);
        element.textContent = `${formatTime(time)} · score ${formatScore(score)}`;
        element.style.left = `${ratio * 100}%`;
        element.classList.add("visible");
      }

      function scoreForSourceTime(sourceTime) {
        return segmentForSource(sourceTime).score;
      }

      function clearTimelinePreview() {
        hoverPreview = null;
        sourcePreview.classList.remove("visible");
        replayPreview.classList.remove("visible");
        drawAll();
      }

      function formatTime(totalSeconds) {
        const seconds = Math.max(0, Math.floor(totalSeconds));
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const remainingSeconds = seconds % 60;
        if (hours > 0) {
          return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
        }
        return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
      }

      function formatScore(score) {
        if (Math.abs(score) >= 100) return score.toFixed(1);
        return score.toFixed(2);
      }

      function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
      }
    </script>
  </body>
</html>
"""
