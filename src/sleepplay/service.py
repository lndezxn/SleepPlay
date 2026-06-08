from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sleepplay.config import AppConfig, OutputConfig, ScoreConfig, SpeedConfig
from sleepplay.pipeline import build_timeline
from sleepplay.progress import ProgressReporter
from sleepplay.renderer import render_replay_video
from sleepplay.timeline import Timeline, write_timeline


@dataclass(frozen=True)
class ProcessingResult:
    timeline_path: Path
    replay_path: Path


@dataclass(frozen=True)
class ProcessingOverrides:
    analysis_width: int | None = None
    preprocess_fps: float | None = None
    preprocess_height: int | None = None
    render_fps: float | None = None
    score_type: str | None = None
    speed_type: str | None = None
    still_score: float | None = None
    motion_score: float | None = None
    min_speed: float | None = None
    max_speed: float | None = None
    sensitivity: float | None = None
    smoothing_window: int | None = None


def generate_timeline(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> Timeline:
    timeline = build_timeline(config, progress_reporter)
    write_timeline(config.output.json, timeline)
    return timeline


def render_video(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> None:
    render_replay_video(config.render, progress_reporter)


def process_replay(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> ProcessingResult:
    generate_timeline(config, progress_reporter)
    render_video(config, progress_reporter)
    return ProcessingResult(
        timeline_path=config.output.json,
        replay_path=config.render.output_video,
    )


def config_for_job(
    config: AppConfig,
    input_path: Path,
    job_dir: Path,
    overrides: ProcessingOverrides | None = None,
) -> AppConfig:
    timeline_path = job_dir / "timeline.json"
    replay_path = job_dir / "replay.mp4"
    video_config = replace(config.video, input=input_path)
    preprocess_config = replace(config.preprocess, output=job_dir / "preprocessed.mp4")
    render_config = replace(
        config.render,
        timeline_json=timeline_path,
        output_video=replay_path,
    )
    score_config = config.score
    speed_config = config.speed

    if overrides is not None:
        if overrides.analysis_width is not None:
            video_config = replace(video_config, analysis_width=overrides.analysis_width)
        if overrides.preprocess_fps is not None:
            preprocess_config = replace(preprocess_config, fps=overrides.preprocess_fps)
        if overrides.preprocess_height is not None:
            preprocess_config = replace(preprocess_config, height=overrides.preprocess_height)
        if overrides.render_fps is not None:
            render_config = replace(render_config, fps=overrides.render_fps)
        score_config = override_score_config(score_config, overrides)
        speed_config = override_speed_config(speed_config, overrides)

    return replace(
        config,
        video=video_config,
        preprocess=preprocess_config,
        output=OutputConfig(json=timeline_path),
        render=render_config,
        score=score_config,
        speed=speed_config,
    )


def override_score_config(
    config: ScoreConfig,
    overrides: ProcessingOverrides,
) -> ScoreConfig:
    if overrides.score_type is None:
        return config
    return replace(config, type=overrides.score_type)


def override_speed_config(
    config: SpeedConfig,
    overrides: ProcessingOverrides,
) -> SpeedConfig:
    return replace(
        config,
        type=overrides.speed_type if overrides.speed_type is not None else config.type,
        still_score=(
            overrides.still_score
            if overrides.still_score is not None
            else config.still_score
        ),
        motion_score=(
            overrides.motion_score
            if overrides.motion_score is not None
            else config.motion_score
        ),
        min_speed=(
            overrides.min_speed if overrides.min_speed is not None else config.min_speed
        ),
        max_speed=(
            overrides.max_speed if overrides.max_speed is not None else config.max_speed
        ),
        sensitivity=(
            overrides.sensitivity
            if overrides.sensitivity is not None
            else config.sensitivity
        ),
        smoothing_window=(
            overrides.smoothing_window
            if overrides.smoothing_window is not None
            else config.smoothing_window
        ),
    )


def write_resolved_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config_to_data(config)), f=path)


def config_to_data(config: AppConfig) -> dict[str, Any]:
    return stringify_paths(asdict(config))


def stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    return value
