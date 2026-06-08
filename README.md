SleepPlay generates replay metadata for sleep videos.

## Usage

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

Available score configs:

```yaml
includes:
  - scores/frame_diff.yaml
  - scores/gradient_diff.yaml
```

`frame_diff` compares pixel differences between adjacent preprocessed frames.
`gradient_diff` compares 3x3 Sobel gradient differences, which is more focused
on edge and shape movement than uniform brightness changes.

The pipeline first preprocesses the input video with the configured ffmpeg
settings, then generates replay metadata from the preprocessed video:

```yaml
preprocess:
  enabled: true
  output: output/preprocessed.mp4
  fps: 1.0
  height: 320
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

Render a replay video from the JSON:

```bash
uv run sleepplay render --config configs/default.yaml
```

The replay video uses the original video as the visual source, applies each
record's `replay_speed`, and writes the current speed in the upper-right corner.
