"""Plotly-based quadrotor trajectory visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _sample_geometry(traj7: Any, ts: float) -> tuple[np.ndarray, ...]:
    """Sample Trajectory7 geometry at uniform time intervals.

    @param[in] traj7  minco.poly_traj.Trajectory7 instance
    @param[in] ts     Sampling interval [s]
    @return           (t, pos, vel, acc, jer, sna) arrays
    """
    duration = traj7.total_duration
    num = max(int(duration / ts) + 1, 2)
    t = np.linspace(0.0, duration, num, dtype=np.float64)
    pos = np.column_stack([np.array(traj7.get_pos(ti), dtype=np.float64).ravel() for ti in t]).T
    vel = np.column_stack([np.array(traj7.get_vel(ti), dtype=np.float64).ravel() for ti in t]).T
    acc = np.column_stack([np.array(traj7.get_acc(ti), dtype=np.float64).ravel() for ti in t]).T
    jer = np.column_stack([np.array(traj7.get_jer(ti), dtype=np.float64).ravel() for ti in t]).T
    sna = np.column_stack([np.array(traj7.get_sna(ti), dtype=np.float64).ravel() for ti in t]).T
    return t, pos, vel, acc, jer, sna


def _box_wireframe(center: np.ndarray, half_extents: tuple[float, float, float]) -> np.ndarray:
    """Return (3, N) array of box edge vertices with NaN separators."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
        ]
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    nan = np.array([np.nan, np.nan, np.nan])
    segs = [np.stack([corners[i], corners[j], nan]) for i, j in edges]
    return np.vstack(segs).T


def visualize(
    traj7: Any,
    output_path: str | Path,
    *,
    ts: float = 0.03,
    title: str | None = None,
    seed_waypoints: np.ndarray | None = None,
    sfc_centers: list[np.ndarray] | None = None,
    half_extents: tuple[float, float, float] | None = None,
    optimized_positions: np.ndarray | None = None,
) -> Path:
    """Generate interactive Plotly HTML visualization of a trajectory.

    @param[in] traj7              minco.poly_traj.Trajectory7 instance
    @param[in] output_path        Output HTML path
    @param[in] ts                 Sampling interval for geometry plots [s]
    @param[in] title              Plot title (default: 'GCOPTER Trajectory')
    @param[in] seed_waypoints     (3, N) seed waypoints (optional)
    @param[in] sfc_centers        List of (3,) SFC box centers (optional)
    @param[in] half_extents       SFC box half-extents [m] (optional)
    @param[in] optimized_positions (3, M) GCOPTER-optimized junction positions (optional)
    @return                       Path to the written HTML file
    """
    t, pos, vel, acc, jer, sna = _sample_geometry(traj7, ts)

    v_norm = np.linalg.norm(vel, axis=1)

    fig = make_subplots(
        rows=5,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scene", "rowspan": 5}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "",
            "Position [m]",
            "Velocity [m/s]",
            "Acceleration [m/s²]",
            "Jerk [m/s³]",
            "Snap [m/s⁴]",
        ),
        vertical_spacing=0.05,
    )

    # --- Left panel: 3D trajectory ---
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(color="royalblue", width=3),
            name="trajectory",
        ),
        row=1,
        col=1,
    )

    if seed_waypoints is not None:
        fig.add_trace(
            go.Scatter3d(
                x=seed_waypoints[0],
                y=seed_waypoints[1],
                z=seed_waypoints[2],
                mode="markers",
                marker=dict(color="orange", size=4, symbol="diamond"),
                name="seed waypoints",
            ),
            row=1,
            col=1,
        )

    if optimized_positions is not None:
        fig.add_trace(
            go.Scatter3d(
                x=optimized_positions[0],
                y=optimized_positions[1],
                z=optimized_positions[2],
                mode="markers",
                marker=dict(color="green", size=8, symbol="diamond"),
                name="optimized waypoints",
            ),
            row=1,
            col=1,
        )

    if sfc_centers is not None and half_extents is not None:
        for i, center in enumerate(sfc_centers):
            wire = _box_wireframe(center, half_extents)
            fig.add_trace(
                go.Scatter3d(
                    x=wire[0],
                    y=wire[1],
                    z=wire[2],
                    mode="lines",
                    line=dict(color="dimgray", width=2),
                    opacity=0.35,
                    showlegend=(i == 0),
                    name="SFC",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter3d(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            z=[pos[0, 2]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="diamond"),
            name="start",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[-1, 0]],
            y=[pos[-1, 1]],
            z=[pos[-1, 2]],
            mode="markers",
            marker=dict(color="red", size=8, symbol="circle"),
            name="end",
        ),
        row=1,
        col=1,
    )

    # --- Row 1: position ---
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 0], name="x", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 1], name="y", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 2], name="z", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )

    # --- Row 2: speed ---
    fig.add_trace(
        go.Scatter(x=t, y=v_norm, name="|v| [m/s]", line=dict(width=1.5)),
        row=2,
        col=2,
    )

    # --- Row 3: acceleration ---
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 0], name="ax", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 1], name="ay", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 2], name="az", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )

    # --- Row 4: jerk ---
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 0], name="jx", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 1], name="jy", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 2], name="jz", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )

    # --- Row 5: snap ---
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 0], name="sx", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 1], name="sy", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 2], name="sz", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=title or "GCOPTER Trajectory",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    fig.update_xaxes(title_text="Time [s]", row=5, col=2)
    fig.update_yaxes(title_text="[m/s]", row=2, col=2)
    fig.update_yaxes(title_text="[m/s²]", row=3, col=2)
    fig.update_yaxes(title_text="[m/s³]", row=4, col=2)
    fig.update_yaxes(title_text="[m/s⁴]", row=5, col=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Visualization saved: {output_path}")
    return output_path
