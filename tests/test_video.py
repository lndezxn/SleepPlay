from pathlib import Path

from sleepplay.video import read_video_frames, video_fps
from video_helpers import write_test_video


def test_read_video_frames_uses_adjacent_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    write_test_video(video_path, fps=2.0, values=(0, 40, 80, 120))

    video_frames = read_video_frames(video_path)

    assert video_frames.frame_interval_seconds == 0.5
    assert [frame.time for frame in video_frames.frames] == [0.0, 0.5, 1.0, 1.5]
    assert len(video_frames.frames) == 4


def test_video_fps_reads_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    write_test_video(video_path, fps=4.0, values=(0, 40, 80, 120))

    assert round(video_fps(video_path)) == 4
