"""Demonstrate waypoint-to-trajectory optimization with GCOPTER.

Supports three trajectory shapes with 10 uniformly-sampled waypoints each:
  line   — straight line with smooth start/stop
  circle — circular loop at constant height
  fig8   — lemniscate (figure-8) at constant height

Visualization: 1×2 figure
  left  — 2D bird's-eye view (BEV) trajectory with SFC box overlays + pos Z
  right — 2×2 grid: velocity, acceleration, jerk, snap norms over time

Usage:
    uv run demo trajectory              # all shapes
    uv run demo trajectory line         # line_traj only
    uv run demo trajectory circle       # circle_traj only
    uv run demo trajectory fig8         # fig8_traj only
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")

SHAPES = ("line", "circle", "fig8")


@dataclass
class Limits:
    """Per-shape velocity / acceleration soft limits for the optimizer."""

    v_max: float = 10.0
    acc_max: float = 10.0

    @classmethod
    def from_yaml(cls, path: str = "config/default_gcopter.yaml") -> Limits:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cost = cfg.get("cost", {})
        return cls(
            v_max=float(cost.get("v_max", 10.0)),
            acc_max=float(cost.get("acc_max", 10.0)),
        )


INTERESTED = (
    ("v_max", "get_max_vel_rate", "[m/s]"),
    ("acc_max", "get_max_acc_rate", "[m/s²]"),
)


def _make_box_planes(size: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -size],
            [-1.0, 0.0, 0.0, -size],
            [0.0, 1.0, 0.0, -size],
            [0.0, -1.0, 0.0, -size],
            [0.0, 0.0, 1.0, -size],
            [0.0, 0.0, -1.0, -size],
        ]
    )


def _center_box(planes: np.ndarray, center: np.ndarray) -> np.ndarray:
    t = planes.copy()
    t[:, 3] -= planes[:, :3] @ center
    return t


def _gen_line() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([10.0, 5.0, 1.0])
    n_waypoints = 10
    t_vectors = np.linspace(start, end, n_waypoints + 1).T  # 3 x (n+1)
    waypoints = t_vectors[:, 1:]  # exclude start
    head_pva = np.column_stack([start, np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pva = np.column_stack([end, np.zeros(3), np.zeros(3), np.zeros(3)])
    total_dist = float(np.linalg.norm(end - start))
    speed = 3.0
    total_time = total_dist / speed
    n_pieces = n_waypoints + 1  # head→wp0, wp0→wp1, ..., wp_{n-1}→tail
    piece_time = np.full(n_pieces, total_time / n_pieces)
    box_size = 0.3
    return head_pva, tail_pva, waypoints, piece_time, box_size, "line_traj"


def _gen_circle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    radius = 5.0
    height = 1.0
    omega = 0.6
    n_waypoints = 10
    total_time = 2.0 * np.pi / omega
    angles = np.linspace(0.0, 2.0 * np.pi, n_waypoints + 2)
    inner_angles = angles[1:-1]
    waypoints = np.vstack(
        [
            radius * np.cos(inner_angles),
            radius * np.sin(inner_angles),
            np.full(inner_angles.shape, height),
        ]
    )
    start_pos = np.array([radius, 0.0, height])
    head_pva = np.column_stack([start_pos, np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pva = head_pva.copy()
    n_pieces = n_waypoints + 1
    piece_time = np.full(n_pieces, total_time / n_pieces)
    box_size = 0.3
    return head_pva, tail_pva, waypoints, piece_time, box_size, "circle_traj"


def _gen_fig8() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    a = 4.0
    height = 0.5
    n_waypoints = 10
    total_time = 12.0
    shift = np.pi / 2
    t_vals = np.linspace(shift, shift + 2.0 * np.pi, n_waypoints + 2)
    inner_t = t_vals[1:-1]
    waypoints = np.vstack(
        [
            a * np.sin(inner_t),
            a * np.sin(inner_t) * np.cos(inner_t),
            np.full(inner_t.shape, height),
        ]
    )
    start_pos = np.array([a * np.sin(shift), a * np.sin(shift) * np.cos(shift), height])
    head_pva = np.column_stack([start_pos, np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pva = head_pva.copy()
    n_pieces = n_waypoints + 1
    piece_time = np.full(n_pieces, total_time / n_pieces)
    box_size = 0.3
    return head_pva, tail_pva, waypoints, piece_time, box_size, "fig8_traj"


def _optimize(
    optimizer: object,
    head_pva: np.ndarray,
    tail_pva: np.ndarray,
    inner_time: np.ndarray,
    inner_points: np.ndarray,
    corridors: list[np.ndarray],
) -> tuple[float, object]:
    ok = optimizer.setup_basic_trajectory(
        head_pva,
        tail_pva,
        inner_time,
        inner_points,
        corridors,
        smoothing_factor=1.0e-1,
        integral_resolution=24,
    )
    if not ok:
        raise RuntimeError("setup_basic_trajectory failed")
    cost, traj = optimizer.optimize(rel_cost_tol=1.0e-3)
    return cost, traj


def _plot_and_save(
    traj: object,
    waypoints: np.ndarray,
    box_size: float,
    label: str,
    limits: Limits,
) -> None:
    import minco

    n_samples = 300
    ts = np.linspace(0.0, traj.total_duration, n_samples)
    positions = np.array([traj.get_pos(t) for t in ts])
    velocities = np.array([traj.get_vel(t) for t in ts])
    accelerations = np.array([traj.get_acc(t) for t in ts])
    jerks = np.array([traj.get_jer(t) for t in ts])
    snaps = np.array([traj.get_sna(t) for t in ts])

    speed = np.linalg.norm(velocities, axis=1)
    acc_norm = np.linalg.norm(accelerations, axis=1)
    jerk_norm = np.linalg.norm(jerks, axis=1)
    snap_norm = np.linalg.norm(snaps, axis=1)

    fig = plt.figure(figsize=(14, 13))
    ax_bev = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
    ax_pos_z = plt.subplot2grid((4, 2), (3, 0))
    ax_vel = plt.subplot2grid((4, 2), (0, 1))
    ax_acc = plt.subplot2grid((4, 2), (1, 1))
    ax_jerk = plt.subplot2grid((4, 2), (2, 1))
    ax_snap = plt.subplot2grid((4, 2), (3, 1))

    box_planes = _make_box_planes(box_size)
    for i in range(waypoints.shape[1]):
        cx, cy = waypoints[0, i], waypoints[1, i]
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
        positions[0, 0], positions[0, 1], c="green", s=120, marker="o", zorder=5, label="Start"
    )
    ax_bev.scatter(
        positions[-1, 0], positions[-1, 1], c="red", s=120, marker="o", zorder=5, label="End"
    )
    ax_bev.scatter(
        waypoints[0, :],
        waypoints[1, :],
        c="orange",
        s=60,
        marker="s",
        zorder=5,
        label="Waypoints",
    )

    junction_ts = np.cumsum(traj.durations)[:-1]
    opt_points = np.array([traj.get_pos(t) for t in junction_ts])
    ax_bev.scatter(
        opt_points[:, 0],
        opt_points[:, 1],
        c="cyan",
        s=40,
        marker="X",
        zorder=6,
        label="Optimized",
    )
    ax_bev.set_xlabel("X [m]")
    ax_bev.set_ylabel("Y [m]")
    ax_bev.set_title(f"BEV — {label}")
    ax_bev.legend(loc="upper left")
    ax_bev.grid(True)
    ax_bev.set_aspect("equal")

    ax_pos_z.plot(ts, positions[:, 2], "m-", linewidth=1.5)
    ax_pos_z.set_xlabel("Time [s]")
    ax_pos_z.set_ylabel("Z [m]")
    ax_pos_z.set_title("Position Z")
    ax_pos_z.grid(True)

    ax_vel.plot(ts, speed, "r-", linewidth=1.5, label="speed")
    ax_vel.axhline(
        y=limits.v_max,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"v_max={limits.v_max:.1f}",
    )
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.set_title("Velocity")
    ax_vel.legend(loc="upper right", fontsize=7)
    ax_vel.grid(True)

    ax_acc.plot(ts, acc_norm, "g-", linewidth=1.5, label="acc")
    ax_acc.axhline(
        y=limits.acc_max,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"acc_max={limits.acc_max:.1f}",
    )
    ax_acc.set_ylabel("Accel [m/s²]")
    ax_acc.set_title("Acceleration")
    ax_acc.legend(loc="upper right", fontsize=7)
    ax_acc.grid(True)

    ax_jerk.plot(ts, jerk_norm, "b-", linewidth=1.5)
    ax_jerk.set_ylabel("Jerk [m/s³]")
    ax_jerk.set_title("Jerk")
    ax_jerk.grid(True)

    ax_snap.plot(ts, snap_norm, "c-", linewidth=1.5)
    ax_snap.set_xlabel("Time [s]")
    ax_snap.set_ylabel("Snap [m/s⁴]")
    ax_snap.set_title("Snap")
    ax_snap.grid(True)

    fig.suptitle(f"GCOPTER — {label}", fontsize=14)
    plt.tight_layout()

    os.makedirs("_tmp", exist_ok=True)
    out_path = f"_tmp/{label}.png"
    fig.savefig(out_path, dpi=120)
    print(f"  Saved figure to {out_path}")
    plt.close(fig)


def run_shape(shape: str) -> None:
    import minco

    if shape == "all":
        for s in SHAPES:
            run_shape(s)
        return

    generators = {
        "line": _gen_line,
        "circle": _gen_circle,
        "fig8": _gen_fig8,
    }
    gen = generators[shape]
    head_pva, tail_pva, waypoints, piece_time, box_size, label = gen()

    box_planes = _make_box_planes(box_size)
    corridors = [_center_box(box_planes, waypoints[:, i]) for i in range(waypoints.shape[1])]

    limits = Limits.from_yaml()

    print(
        f"  {label}: {len(piece_time)} pieces, {waypoints.shape[1]} waypoints, "
        f"total time = {piece_time.sum():.1f}s"
        f"  | v_max={limits.v_max:.0f} acc_max={limits.acc_max:.0f}"
    )

    optimizer = minco.gcopter.GCOPTERPolytopeSFC()
    optimizer.configure_from_file("")

    t0 = time.perf_counter()
    cost, traj = _optimize(optimizer, head_pva, tail_pva, piece_time, waypoints, corridors)
    elapsed = (time.perf_counter() - t0) * 1e3

    actuals = {
        "v_max": getattr(traj, "get_max_vel_rate")(),
        "acc_max": getattr(traj, "get_max_acc_rate")(),
    }

    parts = []
    for key, method, unit in INTERESTED:
        limit = getattr(limits, key)
        actual = actuals[key]
        parts.append(f"{key}={actual:.2f}{unit} (limit={limit:.0f})")
    info = "  ".join(parts)

    print(
        f"  Optimized in {elapsed:.0f} ms  cost={cost:.4f}"
        f"  duration={traj.total_duration:.2f}s  {info}"
    )

    _plot_and_save(traj, waypoints, box_size, label, limits)


def main() -> None:
    print("=== GCOPTER Trajectory Optimization Demos ===\n")
    run_shape("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
