"""NPZ serialization for Trajectory7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(traj7: Any, path: str | Path) -> None:
    """Save a Trajectory7 to NPZ.

    @param[in] traj7  minco.poly_traj.Trajectory7 instance
    @param[in] path   Output .npz path
    """
    durations = np.array(list(traj7.durations), dtype=np.float64)
    coeffs = np.stack([traj7[i].get_coeff_mat() for i in range(len(traj7))])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, durations=durations, coeffs=coeffs)


def load_npz(path: str | Path) -> Any:
    """Reconstruct a Trajectory7 from a .npz coefficient file.

    @param[in] path  Path to trajectory.npz
    @return          minco.poly_traj.Trajectory7 instance
    """
    import minco

    data = np.load(path)
    durations = data["durations"].tolist()
    coeff_mats = [data["coeffs"][i] for i in range(len(durations))]
    return minco.poly_traj.Trajectory7(durations, coeff_mats)
