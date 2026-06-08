import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimelineRecord:
    time: float
    score: float
    replay_speed: float


@dataclass(frozen=True)
class Timeline:
    video: str
    render_video: str
    frame_interval_seconds: float
    records: list[TimelineRecord]


def write_timeline(path: Path, timeline: Timeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(timeline), file, indent=2)
        file.write("\n")


def read_timeline(path: Path) -> Timeline:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Timeline(
        video=str(data["video"]),
        render_video=str(data["render_video"]),
        frame_interval_seconds=float(data["frame_interval_seconds"]),
        records=[
            TimelineRecord(
                time=float(record["time"]),
                score=float(record["score"]),
                replay_speed=float(record["replay_speed"]),
            )
            for record in data["records"]
        ],
    )
