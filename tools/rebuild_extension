#!/usr/bin/env python3
"""Rebuild the minco Python extension in editable mode.

This helper wraps `uv pip install -e . --no-deps`, preserving any extra
arguments supplied on the command line. It is intended for incremental rebuilds
while iterating on the C++ bindings.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _build_command(extra_args: Sequence[str]) -> list[str]:
    project_root = Path(__file__).resolve().parents[1]
    return [
        "uv",
        "pip",
        "install",
        "-e",
        str(project_root),
        "--no-deps",
        *extra_args,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    command = _build_command(args)
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(command, cwd=project_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
