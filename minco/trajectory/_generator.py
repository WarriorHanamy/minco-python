"""Generate feasible quadrotor trajectories via GCOPTER."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minco.trajectory._waypoints import make_sfc_box, waypoints_for_shape

_CONFIG_ROOT = Path(__file__).resolve().parents[1]

_NOMINAL_SPEED = 2.0  # [m/s] initial piece duration seed


@contextmanager
def _in_dir(path: Path):
    old = os.getcwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(old)


@dataclass
class TrajectoryGenerationResult:
    """Result of a GCOPTER trajectory optimization run."""

    cost: float
    traj7: Any
    waypoints: np.ndarray
    sfc_half_extents: tuple[float, float, float]
    sfc_centers: list[np.ndarray]


def generate_trajectory(
    shape: str,
    num_waypoints: int = 10,
    sfc_half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5),
    gcopter_config_path: str | Path | None = None,
    *,
    smoothing_factor: float = 1e-1,
    integral_resolution: int = 24,
    rel_cost_tol: float = 1e-3,
) -> TrajectoryGenerationResult:
    """Run GCOPTER trajectory optimization for a canonical shape.

    @param[in] shape  One of 'circle', 'line', 'fig8'
    @param[in] num_waypoints  Number of waypoints (head + inner + tail)
    @param[in] sfc_half_extents  SFC box half-extents (x, y, z) [m]
    @param[in] gcopter_config_path  Path to GCOPTER config YAML (None uses library default)
    @param[in] smoothing_factor  Corridor smoothness penalty [0=hard, 1=soft]
    @param[in] integral_resolution  Time-integral resolution for corridor penalties
    @param[in] rel_cost_tol  Relative cost tolerance for convergence
    @return  TrajectoryGenerationResult
    """
    import minco

    inner_points = waypoints_for_shape(shape, num_waypoints)
    num_pieces = inner_points.shape[1] + 1

    head_pvaj = np.column_stack([inner_points[:, 0], np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pvaj = np.column_stack([inner_points[:, -1], np.zeros(3), np.zeros(3), np.zeros(3)])

    pts = np.column_stack([inner_points[:, 0], inner_points])
    path_len = 0.0
    for i in range(pts.shape[1] - 1):
        path_len += float(np.linalg.norm(pts[:, i + 1] - pts[:, i]))
    init_dt = max(path_len / num_pieces / _NOMINAL_SPEED, 0.01)
    initial_time = np.full(num_pieces, init_dt)

    sfc_polys = []
    sfc_centers = []
    half_extents = tuple(sfc_half_extents)
    for k in range(num_pieces):
        if k == 0:
            center = inner_points[:, 0]
        elif k == num_pieces - 1:
            center = inner_points[:, -1]
        else:
            center = inner_points[:, k]
        sfc_centers.append(center.copy())
        sfc_polys.append(make_sfc_box(center, half_extents))

    with _in_dir(_CONFIG_ROOT):
        opt = minco.gcopter.GCOPTERPolytopeSFC()

    if gcopter_config_path is not None:
        opt.configure_from_file(str(gcopter_config_path))

    ok = opt.setup_basic_trajectory(
        head_pvaj,
        tail_pvaj,
        initial_time,
        inner_points,
        sfc_polys,
        smoothing_factor=smoothing_factor,
        integral_resolution=integral_resolution,
    )
    if not ok:
        raise RuntimeError("GCOPTER setup_basic_trajectory failed")

    cost, traj7 = opt.optimize(rel_cost_tol=rel_cost_tol)
    return TrajectoryGenerationResult(
        cost=cost,
        traj7=traj7,
        waypoints=inner_points,
        sfc_half_extents=half_extents,
        sfc_centers=sfc_centers,
    )
