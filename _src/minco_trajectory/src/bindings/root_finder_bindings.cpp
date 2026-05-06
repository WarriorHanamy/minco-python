#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include "root_finder.hpp"

namespace py = pybind11;

void bind_root_finder(py::module_ &m)
{
    auto sub = m.def_submodule("root_finder", "Polynomial manipulation and root finding");
    sub.attr("highest_order") = RootFinderParam::highestOrder;
    sub.def("poly_conv", &RootFinder::polyConv, py::arg("lhs"), py::arg("rhs"),
            "Convolve two coefficient vectors (highest degree first).");
    sub.def("poly_sqr", &RootFinder::polySqr, py::arg("coeffs"),
            "Compute the self-convolution of a coefficient vector.");
    sub.def("poly_val", &RootFinder::polyVal, py::arg("coeffs"), py::arg("x"), py::arg("numerical_stability") = true,
            "Evaluate a polynomial at x (coefficients stored highest degree first).");
    sub.def("count_roots", &RootFinder::countRoots, py::arg("coeffs"), py::arg("lbound"), py::arg("ubound"),
            "Count distinct real roots inside (lbound, ubound).");
    sub.def("solve_polynomial",
            [](const Eigen::VectorXd &coeffs, double lbound, double ubound, double tol, bool isolation) {
                auto roots = RootFinder::solvePolynomial(coeffs, lbound, ubound, tol, isolation);
                return std::vector<double>(roots.begin(), roots.end());
            },
            py::arg("coeffs"), py::arg("lbound"), py::arg("ubound"), py::arg("tol"), py::arg("isolation") = true,
            "Compute real roots inside the interval. Returns a sorted list.");
}
