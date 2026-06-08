# Development Fixtures

Generate synthetic videos for debugging SleepPlay scoring and performance:

```bash
uv run python dev/generate_fixtures.py --config dev/fixtures.yaml
```

The generated videos, manifest, and pipeline configs are written under
`data/dev/`, which is ignored by git.

The default config includes short geometry animations for algorithm checks and
1h videos for performance testing. Long videos are generated frame by frame, so
they do not accumulate frames in memory.

Run SleepPlay against a generated performance fixture with:

```bash
uv run sleepplay run --config data/dev/configs/long_periodic_motion_1h.yaml
uv run sleepplay render --config data/dev/configs/long_periodic_motion_1h.yaml
```
