import numpy as np

import minco
import matplotlib
import time

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def visualize_gcopter_trajectory(trajectory, time_samples=None):
    """
    Visualize GCOPTER trajectory with two subplots:
    - Left: 3D trajectory with start (green) and end (red) markers
    - Right: Speed profile over time
    
    Args:
        trajectory: GCOPTER trajectory object
        time_samples: Optional array of time samples, defaults to 500 points
    """
    if time_samples is None:
        time_samples = np.linspace(0.0, trajectory.total_duration, 500)
    
    positions = np.array([trajectory.get_pos(t) for t in time_samples])
    vels = np.array([trajectory.get_vel(t) for t in time_samples])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 5))
    
    # First subplot: 3D trajectory
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=90, azim=0)
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    # Mark start point (green solid ball)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
    # Mark end point (red solid ball) 
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o', label='End')
    ax1.set_title("GCOPTER Trajectory")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend()
    
    # Second subplot: Velocity profile
    ax2 = fig.add_subplot(122)
    speed = np.linalg.norm(vels, axis=1)
    ax2.plot(time_samples, speed, 'r-', linewidth=2)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.set_title("GCOPTER Speed Profile")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def test_gcopter_circle_warm_start_visualization():
    optimizer = minco.gcopter.GCOPTERPolytopeSFC()

    optimizer.configure_flatness(
        mass=1.0,
        gravity=9.81,
        horizontal_drag=0.1,
        vertical_drag=0.1,
        parasitic_drag=0.01,
        speed_smooth=1.0e-3,
        yaw_smooth=1.0e-6,
    )

    optimizer.configure_cost(
        v_max=5.0,
        omg_x_max=1.0,
        omg_y_max=2.0,
        omg_z_max=1.0,
        acc_max=50.0,
        thrust_min=-20.0,
        thrust_max=20.0,
        pos_weight=1.0,
        vel_weight=0.0,
        acc_weight=0.0,
        omg_x_weight=0.0,
        omg_y_weight=0.0,
        omg_z_weight=0.0,
        thrust_weight=0.0,
        time_weight=0.0,
        omg_consistent_weight=0.0,
    )

    radius = 5.0
    height = 1.0
    piece_count = 20

    omega = 0.4
    total_time = 2.0 * np.pi / omega

    pos0 = np.array([radius, 0.0, height])
    vel0 = np.array([0.0, 0.0, 0.0])
    acc0 = np.array([0.0, 0.0, 0.0])

    head_pva = np.column_stack([pos0, vel0, acc0])
    tail_pva = head_pva.copy()

    angles = np.linspace(0.0, 2.0 * np.pi, piece_count + 1)
    inner_angles = angles[1:-1]
    inner_points = np.vstack(
        [
            radius * np.cos(inner_angles),
            radius * np.sin(inner_angles),
            np.full(inner_angles.shape, height),
        ]
    )

    initial_time = np.full(piece_count, total_time / piece_count)

    box_bound = 1.0
    z_max = height + radius * 0.5
    box_planes = np.array(
        [
            [1.0, 0.0, 0.0, -box_bound],
            [-1.0, 0.0, 0.0, -box_bound],
            [0.0, 1.0, 0.0, -box_bound],
            [0.0, -1.0, 0.0, -box_bound],
            [0.0, 0.0, 1.0, -z_max],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    corridors = [box_planes.copy() for _ in range(piece_count - 1)]

    assert optimizer.setup_basic_trajectory(
        head_pva,
        tail_pva,
        initial_time,
        inner_points,
        corridors,
        smoothing_factor=1.0e-2,
        integral_resolution=10,
    )
    start_time = time.time()
    cost, trajectory = optimizer.optimize(rel_cost_tol=1.0e-4)
    end_time = time.time()
    print(f"Optimization took {(end_time - start_time) * 1e3} ms")

    assert np.isfinite(cost)
    assert trajectory.get_piece_num() == piece_count
    print(f"final cost: {cost}")
    time_samples = np.linspace(0.0, trajectory.total_duration, 500)
    positions = np.array([trajectory.get_pos(t) for t in time_samples])
    vels = np.array([trajectory.get_vel(t) for t in time_samples])

    assert np.isfinite(positions).all()
    assert np.linalg.norm(positions[0] - pos0) < 1e-6
    assert np.linalg.norm(positions[-1] - pos0) < 1e-2
    assert np.max(np.linalg.norm(positions, axis=1)) < 1e3

    # Use the separate visualization function
    visualize_gcopter_trajectory(trajectory, time_samples)


if __name__ == "__main__":
    test_gcopter_circle_warm_start_visualization()
