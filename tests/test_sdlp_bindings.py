import numpy as np

import minco


def test_linprog_1d_box_constraints():
    c = np.array([1.0])
    a = np.array([[1.0], [-1.0]])
    b = np.array([1.0, 0.0])

    minimum, argmin = minco.sdlp.linprog(c, a, b)

    assert np.isclose(minimum, 0.0)
    np.testing.assert_allclose(argmin, np.array([0.0]))
