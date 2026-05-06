"""Entry point routing for demo subcommands.

Usage:
    uv run demo flatness              — flatness forward/backward
    uv run demo trajectory            — all trajectory shapes
    uv run demo trajectory line       — line_traj
    uv run demo trajectory circle     — circle_traj
    uv run demo trajectory fig8       — fig8_traj (lemniscate)
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uv run demo", description="Run minco-python demos.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("flatness", help="Demonstrate flatness forward/backward")

    traj_parser = sub.add_parser("trajectory", help="Demonstrate trajectory optimization")
    traj_parser.add_argument(
        "shape",
        nargs="?",
        default="all",
        choices=("line", "circle", "fig8", "all"),
        help="Shape to optimize (default: all)",
    )

    args = parser.parse_args(argv or sys.argv[1:])

    if args.subcommand == "flatness":
        from examples.demo_flatness import main as _flatness

        _flatness()
    elif args.subcommand == "trajectory":
        from examples.demo_trajectory import run_shape as _trajectory

        _trajectory(args.shape)


if __name__ == "__main__":
    main()
