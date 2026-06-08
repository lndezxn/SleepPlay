from sleepplay.config import AppConfig
from sleepplay.preprocess import preprocess_video
from sleepplay.scores.registry import create_scorer
from sleepplay.speeds import SpeedContext, create_speed_mapper
from sleepplay.timeline import Timeline, TimelineRecord
from sleepplay.video import read_video_frames


def build_timeline(config: AppConfig) -> Timeline:
    video_path = preprocess_video(config.video.input, config.preprocess)
    video_frames = read_video_frames(video_path)
    scorer = create_scorer(config.score, config.video)
    speed_mapper = create_speed_mapper(config.speed)

    times: list[float] = []
    scores: list[float] = []
    previous_frame = None
    for video_frame in video_frames.frames:
        if previous_frame is None:
            score = 0.0
        else:
            score = scorer.score(previous_frame, video_frame.frame)

        times.append(video_frame.time)
        scores.append(score)
        previous_frame = video_frame.frame

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

    return Timeline(
        video=str(config.video.input),
        frame_interval_seconds=video_frames.frame_interval_seconds,
        records=records,
    )
