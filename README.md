# SleepPlay

SleepPlay finds motion in long sleep videos, speeds through still periods, slows
down near movement, and renders a replay video with speed metadata.

## Installation

Requirements:

- Python 3.13 or newer.
- [uv](https://docs.astral.sh/uv/) for the local virtual environment and locked
  dependencies.

Set up the project:

```bash
git clone <repo-url>
cd SleepPlay
uv sync
```

Check the install:

```bash
uv run sleepplay --help
uv run python -m pytest
```

The ffmpeg binary used by the pipeline is provided through the locked Python
dependencies, so a separate system ffmpeg install is not required for the
default workflow.

## Configuration

Edit `configs/default.yaml`, then run:

```bash
uv run sleepplay run --config configs/default.yaml
```

`configs/default.yaml` composes separate config blocks:

```yaml
includes:
  - app/default.yaml
  - scores/frame_diff.yaml
  - speeds/sensitive.yaml
```

Swap the score or speed include to combine different algorithms without
duplicating the app/video/render settings.

Available score config includes:

```yaml
scores/frame_diff.yaml
scores/gradient_diff.yaml
```

`frame_diff` compares pixel differences between adjacent preprocessed frames.
`gradient_diff` compares 3x3 Sobel gradient differences, which is more focused
on edge and shape movement than uniform brightness changes.

Available speed config includes:

```yaml
speeds/linear.yaml
speeds/sensitive.yaml
```

`sensitive` is the default because it reacts strongly to small movement and can
smooth the score series before mapping scores to replay speeds.

The pipeline first preprocesses the input video with the configured ffmpeg
settings, then generates replay metadata from the preprocessed video:

```yaml
preprocess:
  enabled: true
  output: output/preprocessed.mp4
  fps: 1.0
  height: 320
```

## Commands

Generate replay metadata:

```bash
uv run sleepplay run --config configs/default.yaml
```

The command writes a JSON file with one record per preprocessed video frame:

```json
{
  "video": "data/input.mp4",
  "frame_interval_seconds": 1.0,
  "records": [
    {
      "time": 0.0,
      "score": 0.0,
      "replay_speed": 16.0
    }
  ]
}
```

The `run` command only generates `time`, `score`, and `replay_speed` metadata.

Render a replay video from the JSON:

```bash
uv run sleepplay render --config configs/default.yaml
```

The replay video uses the original video as the visual source, applies each
record's `replay_speed`, and writes the current speed in the upper-right corner.

Run the local web UI:

```bash
uv run sleepplay web --config configs/default.yaml
```

The web UI uploads videos into `data/web/jobs/`, runs the same pipeline and
renderer as the CLI, streams processing progress, and plays the replay with
source-scale and replay-scale timeline controls. Use the header theme button to
switch between light and dark mode.

## Speed Tuning

Replay speed is intentionally motion-sensitive by default:

```yaml
speed:
  type: sensitive
  still_score: 1.0
  motion_score: 5.0
  sensitivity: 4.0
  smoothing_window: 3
```

Increase `sensitivity` or lower `motion_score` to slow down more aggressively
when small movement appears. Increase `smoothing_window` to smooth noisy scores
before mapping them to replay speed; smoothing uses the whole score series, so a
record's speed can reflect nearby movement.

## Development Fixtures

Generate simple geometry videos for local testing:

```bash
uv run python dev/generate_fixtures.py --config dev/fixtures.yaml
```

The generated videos, web jobs, and fixture outputs live under ignored `data/`
paths and are not intended to be committed.
