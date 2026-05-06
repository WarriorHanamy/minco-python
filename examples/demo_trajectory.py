"""Demonstrate the full waypoint-to-trajectory pipeline with GCOPTER.

Defines waypoints, builds safe-flight-corridor polytopes, runs trajectory
optimization, and visualizes the result as a 1×2 figure:
  left  — 2D bird's-eye view (BEV) trajectory with SFC box overlays
  right — 1×3 grid: velocity, acceleration, jerk norms over time

Usage:
    uv run python examples/demo_trajectory.py
    uv run demo trajectory
"""

from __future__ import annotations

import os
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def main() -> None:
    import minco

    print("=== GCOPTER Trajectory Optimization Demo ===\n")

    optimizer = minco.gcopter.GCOPTERPolytopeSFC()
    optimizer.configure_from_file("")

    head_pva = np.column_stack(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ]
    )
    tail_pva = np.column_stack(
        [
            np.array([8.0, 3.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ]
    )

    inner_points = np.array(
        [
            [2.0, 6.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )

    initial_time = np.array([2.0, 2.0, 2.0])

    box_size = 1.5
    box_planes = np.array(
        [
            [1.0, 0.0, 0.0, -box_size],
            [-1.0, 0.0, 0.0, -box_size],
            [0.0, 1.0, 0.0, -box_size],
            [0.0, -1.0, 0.0, -box_size],
            [0.0, 0.0, 1.0, -box_size],
            [0.0, 0.0, -1.0, -box_size],
        ]
    )

    def _center_box(center: np.ndarray) -> np.ndarray:
        translated = box_planes.copy()
        translated[:, 3] -= box_planes[:, :3] @ center
        return translated

    corridors = [_center_box(point) for point in inner_points.T]

    print(
        f"Problem: {len(initial_time)} pieces, {inner_points.shape[1]} waypoints, "
        f"total time = {initial_time.sum():.0f}s"
    )

    ok = optimizer.setup_basic_trajectory(
        head_pva,
        tail_pva,
        initial_time,
        inner_points,
        corridors,
        smoothing_factor=1.0e-1,
        integral_resolution=24,
    )
    if not ok:
        print("setup_basic_trajectory failed")
        return

    t0 = time.perf_counter()
    cost, traj = optimizer.optimize(rel_cost_tol=1.0e-3)
    elapsed = (time.perf_counter() - t0) * 1e3

    if not np.isfinite(cost):
        print("Optimization did not converge.")
        return

    print(f"Optimized in {elapsed:.0f} ms")
    print(f"Final cost: {cost:.4f}")
    print(f"Total duration: {traj.total_duration:.2f} s")
    print(f"Number of pieces: {traj.get_piece_num()}")

    n_samples = 300
    ts = np.linspace(0.0, traj.total_duration, n_samples)
    positions = np.array([traj.get_pos(t) for t in ts])
    velocities = np.array([traj.get_vel(t) for t in ts])
    accelerations = np.array([traj.get_acc(t) for t in ts])
    jerks = np.array([traj.get_jer(t) for t in ts])

    speed = np.linalg.norm(velocities, axis=1)
    acc_norm = np.linalg.norm(accelerations, axis=1)
    jerk_norm = np.linalg.norm(jerks, axis=1)

    fig = plt.figure(figsize=(14, 10))
    ax_bev = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax_vel = plt.subplot2grid((3, 2), (0, 1))
    ax_acc = plt.subplot2grid((3, 2), (1, 1))
    ax_jerk = plt.subplot2grid((3, 2), (2, 1))

    # --- left: BEV ---
    for i in range(inner_points.shape[1]):
        cx, cy = inner_points[0, i], inner_points[1, i]
        rect = mpatches.Rectangle(
            (cx - box_size, cy - box_size),
            2 * box_size,
            2 * box_size,
            linewidth=0.8,
            edgecolor="gray",
            facecolor="lightblue",
            alpha=0.25,
            zorder=2,
        )
        ax_bev.add_patch(rect)

    ax_bev.plot(positions[:, 0], positions[:, 1], "b-", linewidth=2, label="Trajectory", zorder=4)
    ax_bev.scatter(
        positions[0, 0],
        positions[0, 1],
        c="green",
        s=120,
        marker="o",
        zorder=5,
        label="Start",
    )
    ax_bev.scatter(
        positions[-1, 0],
        positions[-1, 1],
        c="red",
        s=120,
        marker="o",
        zorder=5,
        label="End",
    )
    ax_bev.scatter(
        inner_points[0, :],
        inner_points[1, :],
        c="orange",
        s=60,
        marker="s",
        zorder=5,
        label="Waypoints",
    )
    ax_bev.set_xlabel("X [m]")
    ax_bev.set_ylabel("Y [m]")
    ax_bev.set_title("BEV Trajectory")
    ax_bev.legend(loc="upper left")
    ax_bev.grid(True)
    ax_bev.set_aspect("equal")

    # --- right: 1×3 kinematic profiles ---
    ax_vel.plot(ts, speed, "r-", linewidth=1.5)
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)

    ax_acc.plot(ts, acc_norm, "g-", linewidth=1.5)
    ax_acc.set_ylabel("Accel [m/s²]")
    ax_acc.set_title("Acceleration")
    ax_acc.grid(True)

    ax_jerk.plot(ts, jerk_norm, "b-", linewidth=1.5)
    ax_jerk.set_xlabel("Time [s]")
    ax_jerk.set_ylabel("Jerk [m/s³]")
    ax_jerk.set_title("Jerk")
    ax_jerk.grid(True)

    fig.suptitle("GCOPTER Trajectory Optimization", fontsize=14)
    plt.tight_layout()

    os.makedirs("_tmp", exist_ok=True)
    out_path = "_tmp/demo_trajectory.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
