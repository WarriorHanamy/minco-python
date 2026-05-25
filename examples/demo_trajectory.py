"""Demonstrate trajectory generation with GCOPTER — high-level and low-level API.

Supports three trajectory shapes:
  line   — straight line
  circle — circular loop at constant height
  fig8   — lemniscate (figure-8) at constant height

Usage:
    uv run demo trajectory              # all shapes, high-level API
    uv run demo trajectory line         # line only, low-level API
    uv run demo trajectory circle       # circle only, low-level API
    uv run demo trajectory fig8         # fig8 only, low-level API
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SHAPES = ("line", "circle", "fig8")
_OUT_DIR = Path("_tmp")


def _plot_trajectory(traj, waypoints, label, limits, sfc_size):
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

    for i in range(waypoints.shape[1]):
        cx, cy = waypoints[0, i], waypoints[1, i]
        rect = mpatches.Rectangle(
            (cx - sfc_size, cy - sfc_size),
            2 * sfc_size,
            2 * sfc_size,
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
        waypoints[0], waypoints[1], c="orange", s=60, marker="s", zorder=5, label="Waypoints"
    )

    junction_ts = np.cumsum(traj.durations)[:-1]
    opt_points = np.array([traj.get_pos(t) for t in junction_ts])
    ax_bev.scatter(
        opt_points[:, 0], opt_points[:, 1], c="cyan", s=40, marker="X", zorder=6, label="Optimized"
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
        y=limits[0], color="gray", linestyle="--", linewidth=1.0, label=f"v_max={limits[0]:.1f}"
    )
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.set_title("Velocity")
    ax_vel.legend(loc="upper right", fontsize=7)
    ax_vel.grid(True)

    ax_acc.plot(ts, acc_norm, "g-", linewidth=1.5, label="acc")
    ax_acc.axhline(
        y=limits[1], color="gray", linestyle="--", linewidth=1.0, label=f"acc_max={limits[1]:.1f}"
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

    _OUT_DIR.mkdir(exist_ok=True)
    out_path = _OUT_DIR / f"{label}.png"
    fig.savefig(str(out_path), dpi=120)
    print(f"  Saved figure to {out_path}")
    plt.close(fig)


def _run_low_level(shape):
    import minco
    from minco.trajectory import make_sfc_box, waypoints_for_shape

    waypoints = waypoints_for_shape(shape, num_waypoints=10)
    num_pieces = waypoints.shape[1] + 1
    box_size = 0.3

    head_pva = np.column_stack([waypoints[:, 0], np.zeros(3), np.zeros(3), np.zeros(3)])
    tail_pva = np.column_stack([waypoints[:, -1], np.zeros(3), np.zeros(3), np.zeros(3)])

    pts = np.column_stack([waypoints[:, 0], waypoints])
    path_len = 0.0
    for i in range(pts.shape[1] - 1):
        path_len += float(np.linalg.norm(pts[:, i + 1] - pts[:, i]))
    init_dt = max(path_len / num_pieces / 2.0, 0.01)
    piece_time = np.full(num_pieces, init_dt)

    corridors = [
        make_sfc_box(waypoints[:, min(k, waypoints.shape[1] - 1)], (box_size, box_size, box_size))
        for k in range(num_pieces)
    ]

    label = f"{shape}_traj"
    limits = (10.0, 10.0)
    print(f"  {label}: {len(piece_time)} pieces, {waypoints.shape[1]} waypoints")

    optimizer = minco.gcopter.GCOPTERPolytopeSFC()
    t0 = time.perf_counter()
    ok = optimizer.setup_basic_trajectory(
        head_pva,
        tail_pva,
        piece_time,
        waypoints,
        corridors,
        smoothing_factor=1e-1,
        integral_resolution=24,
    )
    if not ok:
        raise RuntimeError("setup_basic_trajectory failed")
    cost, traj = optimizer.optimize(rel_cost_tol=1e-3)
    elapsed = (time.perf_counter() - t0) * 1e3

    v_max = traj.get_max_vel_rate()
    acc_max = traj.get_max_acc_rate()
    print(
        f"  Optimized in {elapsed:.0f} ms  cost={cost:.4f}  duration={traj.total_duration:.2f}s"
        f"  v_max={v_max:.2f}  acc_max={acc_max:.2f}"
    )

    _plot_trajectory(traj, waypoints, label, limits, box_size)
    return traj


def _run_high_level(shape):
    from minco.trajectory import generate, save_npz, visualize

    result = generate(shape)
    traj = result.traj7
    print(f"  {shape}: cost={result.cost:.4f}  duration={traj.total_duration:.2f}s")

    _OUT_DIR.mkdir(exist_ok=True)
    npz_path = _OUT_DIR / f"{shape}_traj.npz"
    html_path = _OUT_DIR / f"{shape}_traj.html"
    save_npz(traj, npz_path)
    print(f"  Saved NPZ to {npz_path}")
    visualize(
        traj,
        html_path,
        title=f"GCOPTER — {shape}",
        seed_waypoints=result.waypoints,
        sfc_centers=result.sfc_centers,
        half_extents=result.sfc_half_extents,
    )
    return traj


def run_shape(shape):
    if shape == "all":
        for s in SHAPES:
            _run_high_level(s)
        return

    _run_low_level(shape)


def main():
    print("=== GCOPTER Trajectory Optimization Demos ===\n")
    run_shape("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
