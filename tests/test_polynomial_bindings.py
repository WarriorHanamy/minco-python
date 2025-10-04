import numpy as np
import pytest

import minco


@pytest.mark.parametrize("degree", [5, 7])
def test_piece_and_trajectory_roundtrip(degree):
    piece_cls = getattr(minco.poly_traj, f"Piece{degree}")
    traj_cls = getattr(minco.poly_traj, f"Trajectory{degree}")

    coeff_mat = np.ones((3, degree + 1))
    piece = piece_cls(2.0, coeff_mat)

    assert piece.duration == pytest.approx(2.0)
    assert piece.degree == degree
    assert piece.dim == 3
    np.testing.assert_array_equal(piece.get_coeff_mat(), coeff_mat)

    durations = [1.0, 1.5]
    coeffs = [np.random.rand(3, degree + 1) for _ in durations]
    traj = traj_cls(durations, coeffs)

    assert traj.get_piece_num() == len(durations)
    assert traj.total_duration == pytest.approx(sum(durations))
    np.testing.assert_allclose(traj.get_pos(0.5), traj[0].get_pos(0.5))
    np.testing.assert_allclose(traj.get_vel(0.5), traj[0].get_vel(0.5))
    np.testing.assert_allclose(traj.get_acc(0.5), traj[0].get_acc(0.5))

    assert len(list(traj)) == len(durations)
    assert traj.get_max_vel_rate() >= 0.0
    assert traj.get_max_acc_rate() >= 0.0


@pytest.mark.parametrize("degree", [5, 7])
def test_append_piece_updates_duration(degree):
    piece_cls = getattr(minco.poly_traj, f"Piece{degree}")
    traj_cls = getattr(minco.poly_traj, f"Trajectory{degree}")

    base_piece = piece_cls(1.0, np.eye(3, degree + 1))
    traj = traj_cls([base_piece.duration], [base_piece.get_coeff_mat()])

    extra_piece = piece_cls(0.5, np.eye(3, degree + 1))
    original_duration = traj.total_duration
    traj.append_piece(extra_piece)

    assert traj.get_piece_num() == 2
    assert traj.total_duration == pytest.approx(original_duration + extra_piece.duration)
