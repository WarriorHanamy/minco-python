import numpy as np
import minco


def test_trajectory():
    print("Testing Piece5 and Trajectory5 (degree 5 polynomials)...")
    test_degree(5)

    print("\nTesting Piece7 and Trajectory7 (degree 7 polynomials)...")
    test_degree(7)


def test_degree(degree):
    if degree == 5:
        Piece = minco.trajectory.Piece5
        Trajectory = minco.trajectory.Trajectory5
    else:
        Piece = minco.trajectory.Piece7
        Trajectory = minco.trajectory.Trajectory7

    # Test Piece
    print("\nTesting Piece:")
    # Create a coefficient matrix (3x(degree+1))
    coeff_mat = np.ones((3, degree + 1))
    duration = 2.0
    piece = Piece(duration, coeff_mat)

    print(f"Piece duration: {piece.duration}")
    print(f"Piece degree: {piece.degree}")
    print(f"Piece dim: {piece.dim}")
    print(f"Coeff mat shape: {piece.get_coeff_mat().shape}")

    t = 1.0
    print(f"Position at t={t}: {piece.get_pos(t)}")
    print(f"Velocity at t={t}: {piece.get_vel(t)}")
    print(f"Acceleration at t={t}: {piece.get_acc(t)}")
    print(f"Jerk at t={t}: {piece.get_jer(t)}")

    # Test Trajectory
    print("\nTesting Trajectory:")
    # Create multiple pieces
    durations = [1.0, 2.0, 1.5]
    coeff_mats = [np.random.rand(3, degree + 1) for _ in range(3)]
    traj = Trajectory(durations, coeff_mats)

    print(f"Number of pieces: {traj.get_piece_num()}")
    print(f"Total duration: {traj.total_duration}")
    print(f"Durations: {traj.durations}")

    # Test evaluation at different times
    test_times = [0.5, 1.5, 3.0, 4.0]
    for t in test_times:
        print(f"\nAt t={t}:")
        print(f"Position: {traj.get_pos(t)}")
        print(f"Velocity: {traj.get_vel(t)}")
        print(f"Acceleration: {traj.get_acc(t)}")

    # Test piece access
    print("\nTesting piece access:")
    for i, piece in enumerate(traj):
        print(f"Piece {i} duration: {piece.duration}")

    # Test max rates
    print(f"\nMax velocity rate: {traj.get_max_vel_rate()}")
    print(f"Max acceleration rate: {traj.get_max_acc_rate()}")

    # Test appending
    print("\nTesting append:")
    new_piece = Piece(0.5, np.random.rand(3, degree + 1))
    traj.append_piece(new_piece)
    print(f"New piece count: {traj.get_piece_num()}")
    print(f"New total duration: {traj.total_duration}")


def test_sdlp_linprog():
    c = np.array([1.0])
    A = np.array([[1.0], [-1.0]])
    b = np.array([1.0, 0.0])

    minimum, argmin = trajectory.sdlp.linprog(c, A, b)

    assert np.isclose(minimum, 0.0)
    assert np.isclose(argmin[0], 0.0)


def test_root_finder_routines():
    lhs = np.array([1.0, 1.0])
    rhs = np.array([1.0, -1.0])

    conv = trajectory.root_finder.poly_conv(lhs, rhs)
    np.testing.assert_allclose(conv, np.array([1.0, 0.0, -1.0]))

    sqr = trajectory.root_finder.poly_sqr(lhs)
    np.testing.assert_allclose(sqr, np.array([1.0, 2.0, 1.0]))

    value = trajectory.root_finder.poly_val(lhs, 2.0)
    assert np.isclose(value, 3.0)

    count = trajectory.root_finder.count_roots(conv, -1.5, 1.5)
    assert count == 2

    roots = trajectory.root_finder.solve_polynomial(conv, -2.0, 2.0, 1e-6)
    np.testing.assert_allclose(sorted(roots), np.array([-1.0, 1.0]), atol=1e-6)


def test_geo_utils_vertices():
    h_poly = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )

    ok, interior = trajectory.geo_utils.find_interior(h_poly)
    assert ok
    np.testing.assert_allclose(interior, np.array([0.5, 0.5, 0.5]), atol=1e-6)

    vertices = trajectory.geo_utils.enumerate_vertices(h_poly, interior)
    ok_auto, auto_vertices = trajectory.geo_utils.enumerate_vertices_auto(h_poly)
    assert ok_auto
    assert vertices.shape[0] == 3
    assert vertices.shape[1] == 8
    assert auto_vertices.shape == vertices.shape

    obtained = {tuple(np.round(vertices[:, i], 6)) for i in range(vertices.shape[1])}
    expected = {
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    }
    assert obtained == expected
    assert trajectory.geo_utils.overlap(h_poly, h_poly)


def test_flatness_forward_backward():
    mapper = trajectory.flatness.FlatnessMap()
    mapper.reset(
        mass=1.0,
        gravity=9.81,
        horizontal_drag=0.1,
        vertical_drag=0.1,
        parasitic_drag=0.01,
        speed_smooth=1e-3,
    )

    vel = np.zeros(3)
    acc = np.zeros(3)
    jer = np.zeros(3)

    thrust, quat, omg = mapper.forward(vel, acc, jer, psi=0.0, dpsi=0.0)
    np.testing.assert_allclose(thrust, 9.81, atol=1e-6)
    np.testing.assert_allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(omg, np.zeros(3), atol=1e-6)

    grads = mapper.backward(
        pos_grad=np.zeros(3),
        vel_grad=np.zeros(3),
        thr_grad=0.0,
        quat_grad=np.zeros(4),
        omg_grad=np.zeros(3),
    )

    for grad in grads:
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)


if __name__ == "__main__":
    test_trajectory()
