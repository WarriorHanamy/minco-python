import numpy as np

import minco


def test_flatness_forward_backward_zero_motion():
    mapper = minco.flatness.FlatnessMap()
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

    outputs = mapper.backward(
        pos_grad=np.zeros(3),
        vel_grad=np.zeros(3),
        thr_grad=0.0,
        quat_grad=np.zeros(4),
        omg_grad=np.zeros(3),
    )

    for grad in outputs:
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)


if __name__ == "__main__":
    test_flatness_forward_backward_zero_motion()
