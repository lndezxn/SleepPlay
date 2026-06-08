from pathlib import Path


def write_runtime_config(
    path: Path,
    video_path: Path,
    preprocessed_path: Path,
    timeline_path: Path,
    replay_path: Path,
    analysis_width: int = 16,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
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
  fps: 4.0
""",
        encoding="utf-8",
    )
