import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console

from sleepplay.config import load_config
from sleepplay.pipeline import build_timeline
from sleepplay.renderer import render_replay_video
from sleepplay.timeline import write_timeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sleepplay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--config", required=True, type=Path)

    args = parser.parse_args(argv)
    console = Console()

    if args.command == "run":
        config = load_config(args.config)
        timeline = build_timeline(config)
        write_timeline(config.output.json, timeline)
        console.log(f"Wrote timeline to {config.output.json}")
        return 0

    if args.command == "render":
        config = load_config(args.config)
        render_replay_video(config.render)
        console.log(f"Wrote replay video to {config.render.output_video}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")
