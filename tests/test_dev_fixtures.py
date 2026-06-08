import json
from pathlib import Path

import numpy as np

from dev.generate_fixtures import (
    FixtureConfig,
    FixtureDefaults,
    FixtureSpec,
    build_manifest_entry,
    load_fixture_config,
    render_frame,
    write_pipeline_config,
    write_manifest,
)


def test_load_fixture_config(tmp_path: Path) -> None:
    config_path = tmp_path / "fixtures.yaml"
    config_path.write_text(
        """
output_root: data/dev/videos
generate_pipeline_configs: true
pipeline_config_root: data/dev/configs
pipeline_output_root: data/dev/output
defaults:
  width: 64
  height: 36
  fps: 10.0
  codec: mp4v
short_cases:
  - name: still
    kind: still
    duration_seconds: 2
long_cases:
  - name: long_still_1h
    kind: still
    duration_seconds: 3600
""",
        encoding="utf-8",
    )

    config = load_fixture_config(config_path)

    assert config.output_root == Path("data/dev/videos")
    assert config.generate_pipeline_configs is True
    assert len(config.short_cases) == 1
    assert len(config.long_cases) == 1
    assert config.short_cases[0].frame_count == 20
    assert config.long_cases[0].frame_count == 36_000


def test_still_frame_generation_is_repeatable() -> None:
    spec = fixture_spec(kind="still")

    first_frame = render_frame(spec, 0)
    second_frame = render_frame(spec, 5)

    assert first_frame.shape == (36, 64, 3)
    assert np.array_equal(first_frame, second_frame)


def test_motion_frame_generation_changes_over_time() -> None:
    spec = fixture_spec(kind="slow_square")

    first_frame = render_frame(spec, 0)
    later_frame = render_frame(spec, 25)

    assert first_frame.shape == (36, 64, 3)
    assert not np.array_equal(first_frame, later_frame)


def test_write_manifest(tmp_path: Path) -> None:
    spec = fixture_spec(kind="still")
    video_path = tmp_path / "still.mp4"
    manifest_path = tmp_path / "manifest.json"

    write_manifest(manifest_path, [build_manifest_entry(video_path, spec)])

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["fixtures"] == [
        {
            "name": "fixture",
            "group": "short",
            "kind": "still",
            "path": str(video_path),
            "duration_seconds": 2.0,
            "fps": 10.0,
            "width": 64,
            "height": 36,
            "frame_count": 20,
        }
    ]


def test_write_pipeline_config_uses_shared_includes(tmp_path: Path) -> None:
    pipeline_config_root = tmp_path / "configs"
    pipeline_output_root = tmp_path / "output"
    spec = fixture_spec(kind="still")

    write_pipeline_config(
        FixtureConfig(
            output_root=tmp_path / "videos",
            generate_pipeline_configs=True,
            pipeline_config_root=pipeline_config_root,
            pipeline_output_root=pipeline_output_root,
            defaults=FixtureDefaults(width=64, height=36, fps=10.0, codec="mp4v"),
            short_cases=[spec],
            long_cases=[],
        ),
        tmp_path / "videos" / "fixture.mp4",
        spec,
    )

    config_text = (pipeline_config_root / "fixture.yaml").read_text(encoding="utf-8")
    assert "configs/app/default.yaml" in config_text
    assert "configs/scores/frame_diff.yaml" in config_text
    assert "configs/speeds/sensitive.yaml" in config_text
    assert "render_sources/fixture.mp4" in config_text
    assert "score:" not in config_text
    assert "speed:" not in config_text


def fixture_spec(kind: str) -> FixtureSpec:
    return FixtureSpec(
        name="fixture",
        kind=kind,
        group="short",
        duration_seconds=2.0,
        width=64,
        height=36,
        fps=10.0,
        codec="mp4v",
    )
