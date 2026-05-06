"""Public interface for minco-python — wraps the compiled _minco extension."""

import sys as _sys

from ._minco import flatness, gcopter, geo_utils, poly_traj, root_finder, sdlp

_sys.modules["minco.poly_traj"] = poly_traj
_sys.modules["minco.sdlp"] = sdlp
_sys.modules["minco.root_finder"] = root_finder
_sys.modules["minco.geo_utils"] = geo_utils
_sys.modules["minco.flatness"] = flatness
_sys.modules["minco.gcopter"] = gcopter

__all__ = [
    "poly_traj",
    "sdlp",
    "root_finder",
    "geo_utils",
    "flatness",
    "gcopter",
    "flatness_cache",
]
