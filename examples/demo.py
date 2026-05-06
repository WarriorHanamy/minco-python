"""Entry point routing for demo subcommands.

Usage:
    uv run demo flatness     — flatness forward/backward
    uv run demo trajectory   — trajectory optimization + visualization
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uv run demo", description="Run minco-python demos.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("flatness", help="Demonstrate flatness forward/backward")
    sub.add_parser("trajectory", help="Demonstrate waypoint-to-trajectory pipeline")

    args = parser.parse_args(argv or sys.argv[1:])

    if args.subcommand == "flatness":
        from examples.demo_flatness import main as _flatness

        _flatness()
    elif args.subcommand == "trajectory":
        from examples.demo_trajectory import main as _trajectory

        _trajectory()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
