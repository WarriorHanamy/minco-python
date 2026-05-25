"""High-level trajectory generation API.

Composes minco.gcopter, minco.poly_traj, and provides:
- waypoint seeding for canonical shapes
- GCOPTER optimization orchestration
- NPZ serialization
- Plotly HTML visualization
"""

from __future__ import annotations

import sys as _sys

from ._waypoints import SHAPES, make_sfc_box, waypoints_for_shape

_sys.modules["minco.trajectory"] = _sys.modules[__name__]

__all__ = [
    "SHAPES",
    "waypoints_for_shape",
    "make_sfc_box",
    "generate",
    "save_npz",
    "load_npz",
    "visualize",
]


def __getattr__(name: str):
    if name == "generate":
        from ._generator import generate_trajectory as _fn

        return _fn
    if name == "save_npz":
        from ._serialization import save_npz as _fn

        return _fn
    if name == "load_npz":
        from ._serialization import load_npz as _fn

        return _fn
    if name == "visualize":
        from ._plot import visualize as _fn

        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
