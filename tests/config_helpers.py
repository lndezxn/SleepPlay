from pathlib import Path


def write_runtime_config(
    path: Path,
    video_path: Path,
    preprocessed_path: Path,
    timeline_path: Path,
    replay_path: Path,
    render_source_path: Path | None = None,
    analysis_width: int = 16,
    web_storage_root: Path | None = None,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if web_storage_root is None:
        web_storage_root = path.parent / "web" / "jobs"
    if render_source_path is None:
        render_source_path = path.parent / "render_source.mp4"
    path.write_text(
        f"""
includes:
  - {repository_root / "configs/app/default.yaml"}
  - {repository_root / "configs/scores/frame_diff.yaml"}
  - {repository_root / "configs/speeds/sensitive.yaml"}

video:
  input: {video_path}
  analysis_width: {analysis_width}

preprocess:
  output: {preprocessed_path}
  height: {analysis_width}

output:
  json: {timeline_path}

render:
  timeline_json: {timeline_path}
  output_video: {replay_path}
  source_video: {render_source_path}
  source_fps: 2.0

web:
  storage_root: {web_storage_root}
""",
        encoding="utf-8",
    )
