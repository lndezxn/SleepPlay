from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProgressUpdate:
    stage: str
    progress: float
    message: str


class ProgressReporter(Protocol):
    def __call__(self, update: ProgressUpdate) -> None:
        """Receive a progress update for long-running processing work."""


def report_progress(
    reporter: ProgressReporter | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if reporter is None:
        return

    reporter(
        ProgressUpdate(
            stage=stage,
            progress=min(max(progress, 0.0), 1.0),
            message=message,
        )
    )
