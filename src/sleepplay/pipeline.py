from dataclasses import replace
from pathlib import Path

from sleepplay.config import AppConfig, PreprocessConfig
from sleepplay.preprocess import preprocess_video, preprocess_video_pair
from sleepplay.progress import ProgressReporter, report_progress
from sleepplay.scores.registry import create_scorer
from sleepplay.speeds import SpeedContext, create_speed_mapper
from sleepplay.timeline import Timeline, TimelineRecord
from sleepplay.video import read_video_frames


def build_timeline(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> Timeline:
    video_path, render_video_path = preprocess_inputs(config, progress_reporter)
    report_progress(progress_reporter, "timeline", 0.0, "Building timeline")
    video_frames = read_video_frames(
        video_path,
        progress_reporter=progress_reporter,
        progress_start=0.0,
        progress_end=0.5,
    )
    scorer = create_scorer(config.score, config.video)
    speed_mapper = create_speed_mapper(config.speed)

    times: list[float] = []
    scores: list[float] = []
    previous_frame = None
    total_frames = len(video_frames.frames)
    for frame_index, video_frame in enumerate(video_frames.frames, start=1):
        if previous_frame is None:
            score = 0.0
        else:
            score = scorer.score(previous_frame, video_frame.frame)

        times.append(video_frame.time)
        scores.append(score)
        previous_frame = video_frame.frame
        report_progress(
            progress_reporter,
            "timeline",
            0.5 + 0.5 * frame_index / total_frames,
            "Scoring analysis frames",
        )

    replay_speeds = speed_mapper.map_scores(
        SpeedContext(
            times=times,
            scores=scores,
            frame_interval_seconds=video_frames.frame_interval_seconds,
        )
    )

    records: list[TimelineRecord] = []
    for time, score, replay_speed in zip(times, scores, replay_speeds):
        records.append(
            TimelineRecord(
                time=time,
                score=score,
                replay_speed=replay_speed,
            )
        )

    timeline = Timeline(
        video=str(config.video.input),
        render_video=str(render_video_path),
        frame_interval_seconds=video_frames.frame_interval_seconds,
        records=records,
    )
    report_progress(progress_reporter, "timeline", 1.0, "Timeline complete")
    return timeline


def preprocess_inputs(
    config: AppConfig,
    progress_reporter: ProgressReporter | None,
) -> tuple[Path, Path]:
    if config.render.source == "preprocessed" and config.preprocess.enabled:
        render_config = render_preprocess_config(config)
        if render_config.output == config.preprocess.output:
            if render_config.fps != config.preprocess.fps:
                raise ValueError(
                    "render source output must differ from analysis preprocess output "
                    "when frame rates differ."
                )
            analysis_path = preprocess_video(config.video.input, config.preprocess, progress_reporter)
            return analysis_path, analysis_path
        return preprocess_video_pair(
            config.video.input,
            config.preprocess,
            render_config,
            progress_reporter,
            message="Preprocessing analysis and render source",
        )

    analysis_path = preprocess_video(
        config.video.input,
        config.preprocess,
        progress_reporter,
        progress_start=0.0,
        progress_end=0.5 if config.render.source == "preprocessed" else 1.0,
        message="Preprocessing analysis video",
        complete_message="Analysis preprocessing complete",
    )
    return analysis_path, render_source_path(config, analysis_path, progress_reporter)


def render_source_path(
    config: AppConfig,
    analysis_path: Path,
    progress_reporter: ProgressReporter | None,
) -> Path:
    if config.render.source == "original":
        return config.video.input
    if config.render.source == "preprocessed":
        render_config = render_preprocess_config(config)
        if render_config.output == analysis_path:
            if render_config.fps != config.preprocess.fps:
                raise ValueError(
                    "render source output must differ from analysis preprocess output "
                    "when frame rates differ."
                )
            return analysis_path
        return preprocess_video(
            config.video.input,
            render_config,
            progress_reporter,
            progress_start=0.5,
            progress_end=1.0,
            message="Preprocessing render source",
            complete_message="Render source preprocessing complete",
        )
    raise ValueError("render source must be 'original' or 'preprocessed'.")


def render_preprocess_config(config: AppConfig) -> PreprocessConfig:
    if config.render.source_fps <= 0.0:
        raise ValueError("render source fps must be positive.")
    return replace(
        config.preprocess,
        enabled=True,
        output=config.render.source_video,
        fps=config.render.source_fps,
        height=config.preprocess.height,
    )
