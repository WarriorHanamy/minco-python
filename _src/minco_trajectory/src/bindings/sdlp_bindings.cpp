#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include "sdlp.hpp"

namespace py = pybind11;

namespace
{
    template <int Dim>
    std::pair<double, Eigen::VectorXd> linprog_fixed_dim(const Eigen::VectorXd &c,
                                                         const Eigen::MatrixXd &A,
                                                         const Eigen::VectorXd &b)
    {
        Eigen::Matrix<double, Dim, 1> c_fixed = c;
        Eigen::Matrix<double, Eigen::Dynamic, Dim> A_fixed(A.rows(), A.cols());
        A_fixed = A;
        Eigen::Matrix<double, Eigen::Dynamic, 1> b_fixed = b;
        Eigen::Matrix<double, Dim, 1> x_fixed;
        const double minimum = sdlp::linprog<Dim>(c_fixed, A_fixed, b_fixed, x_fixed);
        Eigen::VectorXd x(Dim);
        x = x_fixed;
        return {minimum, x};
    }
}

void bind_sdlp(py::module_ &m)
{
    auto sub = m.def_submodule("sdlp", "Small-dimensional linear programming helpers");
    sub.def("linprog",
            [](const Eigen::VectorXd &c, const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
                if (A.cols() != c.size())
                {
                    throw py::value_error("Coefficient vector dimension must match A columns");
                }
                if (b.size() != A.rows())
                {
                    throw py::value_error("Right-hand side dimension must match A rows");
                }

                const auto dim = static_cast<int>(c.size());
                std::pair<double, Eigen::VectorXd> result;
                switch (dim)
                {
                case 1:
                    result = linprog_fixed_dim<1>(c, A, b);
                    break;
                case 2:
                    result = linprog_fixed_dim<2>(c, A, b);
                    break;
                case 3:
                    result = linprog_fixed_dim<3>(c, A, b);
                    break;
                case 4:
                    result = linprog_fixed_dim<4>(c, A, b);
                    break;
                case 5:
                    result = linprog_fixed_dim<5>(c, A, b);
                    break;
                case 6:
                    result = linprog_fixed_dim<6>(c, A, b);
                    break;
                default:
                    throw py::value_error("linprog currently supports dimensions 1 through 6");
                }

                return py::make_tuple(result.first, result.second);
            },
            py::arg("c"), py::arg("A"), py::arg("b"),
            "Solve min c^T x subject to Ax <= b. Returns (minimum, argmin). Dimensions up to 6 are supported.");
}
