import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console

from sleepplay.config import load_config
from sleepplay.service import generate_timeline, render_video
from sleepplay.web import create_app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sleepplay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--config", required=True, type=Path)

    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--config", required=True, type=Path)

    args = parser.parse_args(argv)
    console = Console()

    if args.command == "run":
        config = load_config(args.config)
        generate_timeline(config)
        console.log(f"Wrote timeline to {config.output.json}")
        return 0

    if args.command == "render":
        config = load_config(args.config)
        render_video(config)
        console.log(f"Wrote replay video to {config.render.output_video}")
        return 0

    if args.command == "web":
        import uvicorn

        config = load_config(args.config)
        uvicorn.run(
            create_app(config),
            host=config.web.host,
            port=config.web.port,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")
