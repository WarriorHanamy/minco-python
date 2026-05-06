#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "bindings.hpp"
#include "gcopter.hpp"
#include "flatness_casadi.hpp"

namespace py = pybind11;

void bind_gcopter(py::module_ &m)
{
    auto sub = m.def_submodule("gcopter", "Trajectory optimization helpers built on GCOPTER");

    using DefaultOptimizer = gcopter::GCOPTER_PolytopeSFC<>;
    using CasadiOptimizer =
        gcopter::GCOPTER_PolytopeSFC<minco::flatness::CasadiQuadrotorFlatnessMap>;

    py::class_<DefaultOptimizer>(sub, "GCOPTERPolytopeSFC")
        .def(py::init<>())
        .def("configure_from_file",
             [](DefaultOptimizer &self, const std::string &file_path) {
                 self.configure_from_file(file_path);
             },
             py::arg("file_path") = std::string(),
             "Load GCOPTER parameters (flatness, cost, and solver tuning) from a YAML configuration file.")
        .def("setup_basic_trajectory",
             &DefaultOptimizer::setup_basic_trajectory,
             py::arg("initial_pva"), py::arg("terminal_pva"),
             py::arg("initial_time"), py::arg("initial_points"),
             py::arg("sfc_control_points"), py::arg("smoothing_factor"),
             py::arg("integral_resolution"))
        .def("optimize",
             [](DefaultOptimizer &self, double rel_cost_tol) {
                 Trajectory<5> traj;
                 double       cost = self.optimize(traj, rel_cost_tol);
                 return py::make_tuple(cost, traj);
             },
             py::arg("rel_cost_tol"));

    py::class_<CasadiOptimizer>(sub, "GCOPTERPolytopeSFCCasadi")
        .def(py::init<>())
        .def(
            "configure_from_file",
            [](CasadiOptimizer &self, const std::string &file_path) {
                self.configure_from_file(file_path);
            },
            py::arg("file_path") = std::string(),
            "Load GCOPTER parameters when using the CasADi quadrotor flatness map. "
            "Flatness parameters are embedded at build time; re-run the CasADi code "
            "generation tool to update them.")
        .def("setup_basic_trajectory",
             &CasadiOptimizer::setup_basic_trajectory,
             py::arg("initial_pva"), py::arg("terminal_pva"),
             py::arg("initial_time"), py::arg("initial_points"),
             py::arg("sfc_control_points"), py::arg("smoothing_factor"),
             py::arg("integral_resolution"))
        .def(
            "optimize",
            [](CasadiOptimizer &self, double rel_cost_tol) {
                Trajectory<5> traj;
                double        cost = self.optimize(traj, rel_cost_tol);
                return py::make_tuple(cost, traj);
            },
            py::arg("rel_cost_tol"));
}
