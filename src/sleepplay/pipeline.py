from sleepplay.config import AppConfig
from sleepplay.preprocess import preprocess_video
from sleepplay.progress import ProgressReporter, report_progress
from sleepplay.scores.registry import create_scorer
from sleepplay.speeds import SpeedContext, create_speed_mapper
from sleepplay.timeline import Timeline, TimelineRecord
from sleepplay.video import read_video_frames


def build_timeline(
    config: AppConfig,
    progress_reporter: ProgressReporter | None = None,
) -> Timeline:
    video_path = preprocess_video(config.video.input, config.preprocess, progress_reporter)
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
        frame_interval_seconds=video_frames.frame_interval_seconds,
        records=records,
    )
    report_progress(progress_reporter, "timeline", 1.0, "Timeline complete")
    return timeline
