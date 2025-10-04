import numpy as np

import minco


def test_polynomial_helpers_and_roots():
    lhs = np.array([1.0, 1.0])
    rhs = np.array([1.0, -1.0])

    conv = minco.root_finder.poly_conv(lhs, rhs)
    np.testing.assert_allclose(conv, np.array([1.0, 0.0, -1.0]))

    rhs_sqr = minco.root_finder.poly_sqr(rhs)
    np.testing.assert_allclose(rhs_sqr, np.array([1.0, -2.0, 1.0]))

    lhs_sqr = minco.root_finder.poly_sqr(lhs)
    np.testing.assert_allclose(lhs_sqr, np.array([1.0, 2.0, 1.0]))

    value = minco.root_finder.poly_val(lhs, 2.0)
    assert np.isclose(value, 3.0)

    another = np.array([2.0, 0.0, -3.0])
    value2 = minco.root_finder.poly_val(another, 2.0)
    assert np.isclose(value2, 5.0)

    count = minco.root_finder.count_roots(conv, -1.5, 1.5)
    assert count == 2

    count = minco.root_finder.count_roots(rhs_sqr, -1.5, 0.5)
    assert count == 0

    roots = minco.root_finder.solve_polynomial(conv, -2.0, 2.0, tol=1e-6)
    np.testing.assert_allclose(sorted(roots), np.array([-1.0, 1.0]), atol=1e-6)

    roots = minco.root_finder.solve_polynomial(conv, -2.0, 0.0, tol=1e-6)
    np.testing.assert_allclose(sorted(roots), np.array([-1.0]), atol=1e-6)


if __name__ == "__main__":
    test_polynomial_helpers_and_roots()
