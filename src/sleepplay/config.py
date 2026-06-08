from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class VideoConfig:
    input: Path
    analysis_width: int


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool
    output: Path
    fps: float
    height: int
    video_codec: str
    preset: str
    crf: int
    pixel_format: str
    overwrite: bool


@dataclass(frozen=True)
class OutputConfig:
    json: Path


@dataclass(frozen=True)
class RenderOverlayConfig:
    enabled: bool
    margin: int
    font_scale: float
    thickness: int


@dataclass(frozen=True)
class RenderConfig:
    timeline_json: Path
    output_video: Path
    fps: float
    video_codec: str
    quality: int
    overlay: RenderOverlayConfig


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int
    storage_root: Path
    max_workers: int


@dataclass(frozen=True)
class ScoreConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeedConfig:
    type: str
    still_score: float
    motion_score: float
    min_speed: float
    max_speed: float
    sensitivity: float
    smoothing_window: int


@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig
    preprocess: PreprocessConfig
    output: OutputConfig
    render: RenderConfig
    web: WebConfig
    score: ScoreConfig
    speed: SpeedConfig


def load_config(path: Path) -> AppConfig:
    raw_config = load_composed_config(path)
    data = OmegaConf.to_container(raw_config, resolve=True, throw_on_missing=True)

    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping.")

    video = data["video"]
    preprocess = data["preprocess"]
    output = data["output"]
    render = data["render"]
    web = data["web"]
    score = data["score"]
    speed = data["speed"]

    return AppConfig(
        video=VideoConfig(
            input=Path(str(video["input"])),
            analysis_width=int(video["analysis_width"]),
        ),
        preprocess=PreprocessConfig(
            enabled=bool(preprocess["enabled"]),
            output=Path(str(preprocess["output"])),
            fps=float(preprocess["fps"]),
            height=int(preprocess["height"]),
            video_codec=str(preprocess["video_codec"]),
            preset=str(preprocess["preset"]),
            crf=int(preprocess["crf"]),
            pixel_format=str(preprocess["pixel_format"]),
            overwrite=bool(preprocess["overwrite"]),
        ),
        output=OutputConfig(json=Path(str(output["json"]))),
        render=RenderConfig(
            timeline_json=Path(str(render["timeline_json"])),
            output_video=Path(str(render["output_video"])),
            fps=float(render["fps"]),
            video_codec=str(render["video_codec"]),
            quality=int(render["quality"]),
            overlay=RenderOverlayConfig(
                enabled=bool(render["overlay"]["enabled"]),
                margin=int(render["overlay"]["margin"]),
                font_scale=float(render["overlay"]["font_scale"]),
                thickness=int(render["overlay"]["thickness"]),
            ),
        ),
        web=WebConfig(
            host=str(web["host"]),
            port=int(web["port"]),
            storage_root=Path(str(web["storage_root"])),
            max_workers=int(web["max_workers"]),
        ),
        score=ScoreConfig(
            type=str(score["type"]),
            params=dict(score.get("params", {})),
        ),
        speed=SpeedConfig(
            type=str(speed["type"]),
            still_score=float(speed["still_score"]),
            motion_score=float(speed["motion_score"]),
            min_speed=float(speed["min_speed"]),
            max_speed=float(speed["max_speed"]),
            sensitivity=float(speed["sensitivity"]),
            smoothing_window=int(speed["smoothing_window"]),
        ),
    )


def load_composed_config(path: Path) -> DictConfig:
    raw_config = OmegaConf.load(path)
    include_paths = read_include_paths(raw_config)
    if not include_paths:
        return raw_config

    included_configs = [
        load_composed_config(resolve_include_path(path.parent, include_path))
        for include_path in include_paths
    ]
    local_config = OmegaConf.create(
        {
            key: value
            for key, value in OmegaConf.to_container(raw_config, resolve=False).items()
            if key != "includes"
        }
    )
    return OmegaConf.merge(*included_configs, local_config)


def read_include_paths(config: DictConfig) -> list[str]:
    if "includes" not in config:
        return []

    includes = OmegaConf.to_container(config["includes"], resolve=True)
    if isinstance(includes, list):
        return [str(include_path) for include_path in includes]
    if isinstance(includes, dict):
        return [str(include_path) for include_path in includes.values()]

    raise TypeError("includes must be a list or mapping of config paths.")


def resolve_include_path(config_dir: Path, include_path: str) -> Path:
    path = Path(include_path)
    if path.is_absolute():
        return path
    return config_dir / path
