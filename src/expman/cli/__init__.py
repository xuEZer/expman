"""Console entry point for managing experiment batches."""

import sys

from expman.cli.commands import build_parser
from expman.settings import project_settings


def main(argv: list[str] | None = None) -> int:
    """Run the requested command and return its exit status."""
    if argv is None:
        argv = sys.argv[1:]
    with project_settings():
        args = build_parser().parse_args(argv)
        if args.command in ("run", "resume"):
            from expman.cli._background import launch_background

            return launch_background(argv)
        return args.handler(args)
