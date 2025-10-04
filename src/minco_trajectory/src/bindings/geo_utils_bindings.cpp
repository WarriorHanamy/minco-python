#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include "geo_utils.hpp"

namespace py = pybind11;

namespace
{
    Eigen::MatrixX4d require_hpoly4(const Eigen::MatrixXd &h_poly)
    {
        if (h_poly.cols() != 4)
        {
            throw py::value_error("Expected matrix with 4 columns for half-space polytope");
        }
        return h_poly;
    }
}

void bind_geo_utils(py::module_ &m)
{
    auto sub = m.def_submodule("geo_utils", "Half-space polytope helpers");
    sub.def("find_interior",
            [](const Eigen::MatrixXd &h_poly) {
                Eigen::MatrixX4d h = require_hpoly4(h_poly);
                Eigen::Vector3d interior;
                const bool ok = geo_utils::findInterior(h, interior);
                return py::make_tuple(ok, interior);
            },
            py::arg("h_poly"),
            "Locate an interior point of a convex polytope defined by half-spaces.");
    sub.def("overlap",
            [](const Eigen::MatrixXd &left, const Eigen::MatrixXd &right, double epsilon) {
                return geo_utils::overlap(require_hpoly4(left), require_hpoly4(right), epsilon);
            },
            py::arg("left"), py::arg("right"), py::arg("epsilon") = 1.0e-6,
            "Check whether two half-space polytopes overlap.");
    sub.def("enumerate_vertices",
            [](const Eigen::MatrixXd &h_poly, const Eigen::Vector3d &interior, double epsilon) {
                Eigen::Matrix3Xd vertices;
                geo_utils::enumerateVs(require_hpoly4(h_poly), interior, vertices, epsilon);
                return vertices;
            },
            py::arg("h_poly"), py::arg("interior"), py::arg("epsilon") = 1.0e-6,
            "Enumerate vertices given half-spaces and a known interior point.");
    sub.def("enumerate_vertices_auto",
            [](const Eigen::MatrixXd &h_poly, double epsilon) {
                Eigen::Matrix3Xd vertices;
                const bool ok = geo_utils::enumerateVs(require_hpoly4(h_poly), vertices, epsilon);
                return py::make_tuple(ok, vertices);
            },
            py::arg("h_poly"), py::arg("epsilon") = 1.0e-6,
            "Enumerate vertices by first finding an interior point. Returns (success, vertices).");
}
