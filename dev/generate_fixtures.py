import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress


@dataclass(frozen=True)
class FixtureDefaults:
    width: int
    height: int
    fps: float
    codec: str


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    kind: str
    group: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str

    @property
    def frame_count(self) -> int:
        return round(self.duration_seconds * self.fps)


@dataclass(frozen=True)
class FixtureConfig:
    output_root: Path
    generate_pipeline_configs: bool
    pipeline_config_root: Path
    pipeline_output_root: Path
    defaults: FixtureDefaults
    short_cases: list[FixtureSpec]
    long_cases: list[FixtureSpec]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_fixtures")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    console = Console()
    config = load_fixture_config(args.config)
    generated = generate_fixtures(config)
    console.log(f"Wrote {len(generated)} fixture videos under {config.output_root}")
    return 0


def load_fixture_config(path: Path) -> FixtureConfig:
    raw_config = OmegaConf.load(path)
    data = OmegaConf.to_container(raw_config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError("Fixture config root must be a mapping.")

    defaults = FixtureDefaults(
        width=int(data["defaults"]["width"]),
        height=int(data["defaults"]["height"]),
        fps=float(data["defaults"]["fps"]),
        codec=str(data["defaults"]["codec"]),
    )

    short_cases = [
        parse_case(case_data, "short", defaults) for case_data in data["short_cases"]
    ]
    long_cases = [
        parse_case(case_data, "long", defaults) for case_data in data["long_cases"]
    ]

    return FixtureConfig(
        output_root=Path(str(data["output_root"])),
        generate_pipeline_configs=bool(data["generate_pipeline_configs"]),
        pipeline_config_root=Path(str(data["pipeline_config_root"])),
        pipeline_output_root=Path(str(data["pipeline_output_root"])),
        defaults=defaults,
        short_cases=short_cases,
        long_cases=long_cases,
    )


def parse_case(
    case_data: dict[str, object],
    group: str,
    defaults: FixtureDefaults,
) -> FixtureSpec:
    return FixtureSpec(
        name=str(case_data["name"]),
        kind=str(case_data["kind"]),
        group=group,
        duration_seconds=float(case_data["duration_seconds"]),
        width=int(case_data.get("width", defaults.width)),
        height=int(case_data.get("height", defaults.height)),
        fps=float(case_data.get("fps", defaults.fps)),
        codec=str(case_data.get("codec", defaults.codec)),
    )


def generate_fixtures(config: FixtureConfig) -> list[dict[str, object]]:
    specs = [*config.short_cases, *config.long_cases]
    manifest_entries: list[dict[str, object]] = []

    with Progress() as progress:
        for spec in specs:
            video_path = fixture_video_path(config.output_root, spec)
            write_fixture_video(video_path, spec, progress)
            manifest_entries.append(build_manifest_entry(video_path, spec))
            if config.generate_pipeline_configs:
                write_pipeline_config(config, video_path, spec)

    write_manifest(config.output_root / "manifest.json", manifest_entries)
    return manifest_entries


def fixture_video_path(output_root: Path, spec: FixtureSpec) -> Path:
    return output_root / spec.group / f"{spec.name}.mp4"


def write_fixture_video(path: Path, spec: FixtureSpec, progress: Progress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*spec.codec)
    writer = cv2.VideoWriter(str(path), fourcc, spec.fps, (spec.width, spec.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create fixture video: {path}")

    task = progress.add_task(f"Writing {path.name}", total=spec.frame_count)
    try:
        for frame_index in range(spec.frame_count):
            writer.write(render_frame(spec, frame_index))
            progress.advance(task)
    finally:
        writer.release()


def render_frame(spec: FixtureSpec, frame_index: int) -> np.ndarray:
    frame = base_frame(spec)
    time_seconds = frame_index / spec.fps

    if spec.kind == "still":
        return frame
    if spec.kind == "slow_square":
        draw_square(frame, x_ratio=loop_ratio(time_seconds / 10.0), size_ratio=0.14)
        return frame
    if spec.kind == "fast_square":
        draw_square(frame, x_ratio=loop_ratio(time_seconds / 2.0), size_ratio=0.14)
        return frame
    if spec.kind == "burst_motion":
        draw_burst_motion(frame, spec, time_seconds)
        return frame
    if spec.kind == "size_change":
        draw_size_change(frame, spec, time_seconds)
        return frame
    if spec.kind == "scene_noise":
        return add_scene_noise(frame, frame_index)
    if spec.kind == "periodic_motion":
        draw_periodic_motion(frame, spec, time_seconds)
        return frame
    if spec.kind == "continuous_motion":
        draw_square(frame, x_ratio=loop_ratio(time_seconds / 4.0), size_ratio=0.18)
        draw_circle(frame, y_ratio=loop_ratio(time_seconds / 7.0), radius_ratio=0.08)
        return frame

    raise ValueError(f"Unknown fixture kind: {spec.kind}")


def base_frame(spec: FixtureSpec) -> np.ndarray:
    frame = np.full((spec.height, spec.width, 3), 24, dtype=np.uint8)
    bed_margin_x = round(spec.width * 0.16)
    bed_margin_y = round(spec.height * 0.24)
    bed_width = spec.width - bed_margin_x * 2
    bed_height = spec.height - bed_margin_y * 2
    cv2.rectangle(
        frame,
        (bed_margin_x, bed_margin_y),
        (bed_margin_x + bed_width, bed_margin_y + bed_height),
        (44, 46, 58),
        -1,
    )
    cv2.rectangle(
        frame,
        (bed_margin_x, bed_margin_y),
        (bed_margin_x + bed_width, bed_margin_y + bed_height),
        (94, 98, 120),
        2,
    )
    cv2.rectangle(
        frame,
        (round(spec.width * 0.22), round(spec.height * 0.28)),
        (round(spec.width * 0.42), round(spec.height * 0.42)),
        (78, 82, 104),
        -1,
    )
    return frame


def draw_square(frame: np.ndarray, x_ratio: float, size_ratio: float) -> None:
    height, width = frame.shape[:2]
    size = round(min(width, height) * size_ratio)
    x = round((width - size) * x_ratio)
    y = round(height * 0.56 - size / 2)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (168, 198, 255), -1)


def draw_circle(frame: np.ndarray, y_ratio: float, radius_ratio: float) -> None:
    height, width = frame.shape[:2]
    radius = round(min(width, height) * radius_ratio)
    center = (round(width * 0.62), round(radius + (height - 2 * radius) * y_ratio))
    cv2.circle(frame, center, radius, (244, 182, 95), -1)


def draw_burst_motion(frame: np.ndarray, spec: FixtureSpec, time_seconds: float) -> None:
    burst_start = spec.duration_seconds * 0.42
    burst_end = spec.duration_seconds * 0.58
    if burst_start <= time_seconds <= burst_end:
        burst_ratio = (time_seconds - burst_start) / (burst_end - burst_start)
        draw_square(frame, x_ratio=loop_ratio(burst_ratio * 2.0), size_ratio=0.2)
    else:
        draw_square(frame, x_ratio=0.5, size_ratio=0.2)


def draw_size_change(frame: np.ndarray, spec: FixtureSpec, time_seconds: float) -> None:
    wave = 0.5 + 0.5 * np.sin(time_seconds * np.pi * 2.0 / spec.duration_seconds)
    radius_ratio = 0.06 + 0.09 * float(wave)
    draw_circle(frame, y_ratio=0.52, radius_ratio=radius_ratio)


def draw_periodic_motion(
    frame: np.ndarray,
    spec: FixtureSpec,
    time_seconds: float,
) -> None:
    cycle_seconds = 300.0
    active_seconds = 20.0
    cycle_time = time_seconds % cycle_seconds
    if cycle_time < active_seconds:
        draw_square(frame, x_ratio=loop_ratio(cycle_time / active_seconds), size_ratio=0.2)
    else:
        draw_square(frame, x_ratio=0.5, size_ratio=0.2)


def add_scene_noise(frame: np.ndarray, frame_index: int) -> np.ndarray:
    generator = np.random.default_rng(frame_index)
    noise = generator.integers(-4, 5, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def loop_ratio(value: float) -> float:
    phase = value % 1.0
    if phase <= 0.5:
        return phase * 2.0
    return 2.0 - phase * 2.0


def build_manifest_entry(path: Path, spec: FixtureSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "group": spec.group,
        "kind": spec.kind,
        "path": str(path),
        "duration_seconds": spec.duration_seconds,
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "frame_count": spec.frame_count,
    }


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"fixtures": entries}
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def write_pipeline_config(
    config: FixtureConfig,
    video_path: Path,
    spec: FixtureSpec,
) -> None:
    config.pipeline_config_root.mkdir(parents=True, exist_ok=True)
    pipeline_config = {
        "includes": [
            relative_config_path(
                config.pipeline_config_root,
                Path("configs/app/default.yaml"),
            ),
            relative_config_path(
                config.pipeline_config_root,
                Path("configs/scores/frame_diff.yaml"),
            ),
            relative_config_path(
                config.pipeline_config_root,
                Path("configs/speeds/sensitive.yaml"),
            ),
        ],
        "video": {
            "input": str(video_path),
            "analysis_width": 320,
        },
        "preprocess": {
            "output": str(config.pipeline_output_root / "preprocessed" / f"{spec.name}.mp4"),
        },
        "output": {
            "json": str(config.pipeline_output_root / "timelines" / f"{spec.name}.json"),
        },
        "render": {
            "timeline_json": str(
                config.pipeline_output_root / "timelines" / f"{spec.name}.json"
            ),
            "output_video": str(config.pipeline_output_root / "replays" / f"{spec.name}.mp4"),
        },
    }
    OmegaConf.save(
        config=OmegaConf.create(pipeline_config),
        f=config.pipeline_config_root / f"{spec.name}.yaml",
    )


def relative_config_path(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir)


if __name__ == "__main__":
    raise SystemExit(main())
