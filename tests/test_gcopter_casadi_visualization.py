from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import minco
import numpy as np
import pytest


def generate_trajectory(shape_type="circle", radius=30.0, height=1.5, piece_count=20):
    pos0 = np.array([radius, 0.0, height])
    vel0 = np.zeros(3)
    acc0 = np.zeros(3)
    head_pva = np.column_stack([pos0, vel0, acc0])
    tail_pva = np.column_stack([pos0 + np.array([1, 0, 0]), vel0, acc0])

    if shape_type == "circle":
        angles = np.linspace(0.0, 2.0 * np.pi, piece_count + 1)
        inner_points = np.vstack(
            [
                radius * np.cos(angles[1:-1]),
                radius * np.sin(angles[1:-1]),
                np.full(piece_count - 1, height),
            ]
        )
    elif shape_type == "figure8":
        t = np.linspace(0, 2 * np.pi, piece_count + 1)[1:-1]
        scale = radius / 2
        inner_points = np.vstack(
            [
                scale * np.sqrt(2) * np.cos(t) / (np.sin(t) ** 2 + 1),
                scale * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1),
                np.full(piece_count - 1, height),
            ]
        )
    elif shape_type == "square":
        points_per_side = piece_count // 4
        side_length = radius
        x = np.linspace(-side_length, side_length, points_per_side + 1)[:-1]
        y = np.linspace(-side_length, side_length, points_per_side + 1)[:-1]

        top = np.vstack([x, np.full_like(x, side_length), np.full_like(x, height)])
        right = np.vstack(
            [np.full_like(y, side_length), y[::-1], np.full_like(y, height)]
        )
        bottom = np.vstack(
            [x[::-1], np.full_like(x, -side_length), np.full_like(x, height)]
        )
        left = np.vstack([np.full_like(y, -side_length), y, np.full_like(y, height)])

        inner_points = np.hstack([top, right, bottom, left])[:, : piece_count - 1]
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    print(f"Generated {shape_type} trajectory points:")
    print(inner_points)

    omega = 0.5
    horizon = 2.0 * np.pi / omega
    initial_time = np.full(piece_count, horizon / piece_count)

    return head_pva, tail_pva, inner_points, initial_time


def visualize_gcopter_trajectory(
    trajectory: Any, time_samples: Optional[np.ndarray] = None
) -> None:
    """Render a GCOPTER trajectory and its speed profile for quick inspection."""
    if time_samples is None:
        time_samples = np.linspace(0.0, trajectory.total_duration, 600)

    positions = np.array([trajectory.get_pos(t) for t in time_samples])
    velocities = np.array([trajectory.get_vel(t) for t in time_samples])
    speeds = np.linalg.norm(velocities, axis=1)

    fig = plt.figure(figsize=(12, 5))

    ax_3d = fig.add_subplot(121, projection="3d")
    ax_3d.view_init(elev=90, azim=45)
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2)
    ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="green", s=60)
    ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c="red", s=60)
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.set_title("CasADi GCOPTER Trajectory")

    ax_speed = fig.add_subplot(122)
    ax_speed.plot(time_samples, speeds, "r-", linewidth=2)
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title("Speed Profile")
    ax_speed.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _run_casadi_gcopter_circle() -> Dict[str, Any]:
    optimizer = minco.gcopter.GCOPTERPolytopeSFCCasadi()
    optimizer.configure_from_file("config/default_gcopter.yaml")
    config = {
        "shape_type": "square",  # 可选: "circle", "figure8", "square"
        "radius": 30.0,
        "height": 1.5,
        "piece_count": 20,
    }

    head_pva, tail_pva, inner_points, initial_time = generate_trajectory(**config)

    corridor_half = 1.0
    z_max = config["height"] + 0.5
    box_planes = np.array(
        [
            [1.0, 0.0, 0.0, -corridor_half],
            [-1.0, 0.0, 0.0, -corridor_half],
            [0.0, 1.0, 0.0, -corridor_half],
            [0.0, -1.0, 0.0, -corridor_half],
            [0.0, 0.0, 1.0, -z_max],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )

    def _center_box_planes(center: np.ndarray) -> np.ndarray:
        translated = box_planes.copy()
        translated[:, 3] -= box_planes[:, :3] @ center
        return translated

    corridors = [_center_box_planes(point) for point in inner_points.T]

    assert optimizer.setup_basic_trajectory(
        head_pva,
        tail_pva,
        initial_time,
        inner_points,
        corridors,
        smoothing_factor=5.0e-3,
        integral_resolution=30,
    )

    cost, trajectory = optimizer.optimize(rel_cost_tol=1.0e-4)
    samples = np.linspace(0.0, trajectory.total_duration, 600)
    positions = np.array([trajectory.get_pos(t) for t in samples])

    assert np.isfinite(cost)
    assert np.isfinite(positions).all()

    return {
        "cost": cost,
        "trajectory": trajectory,
        "time_samples": samples,
        "positions": positions,
    }


def test_gcopter_casadi_circle_visual_debug() -> None:
    pytest.skip("Visualization harness; run module directly for plots.")


if __name__ == "__main__":
    result = _run_casadi_gcopter_circle()
    print(f"CasADi GCOPTER cost: {result['cost']:.6f}")
    visualize_gcopter_trajectory(result["trajectory"], result["time_samples"])
