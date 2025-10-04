#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include "flatness.hpp"

namespace py = pybind11;

void bind_flatness(py::module_ &m)
{
    auto sub = m.def_submodule("flatness", "Forward and backward flatness transforms");
    py::class_<flatness::FlatnessMap>(sub, "FlatnessMap")
        .def(py::init<>())
        .def("reset", &flatness::FlatnessMap::reset,
             py::arg("mass"), py::arg("gravity"), py::arg("horizontal_drag"), py::arg("vertical_drag"),
             py::arg("parasitic_drag"), py::arg("speed_smooth"),
             "Configure vehicle and drag parameters.")
        .def("forward",
             [](flatness::FlatnessMap &self,
                const Eigen::Vector3d &vel,
                const Eigen::Vector3d &acc,
                const Eigen::Vector3d &jer,
                double psi,
                double dpsi) {
                 double thrust = 0.0;
                 Eigen::Vector4d quat;
                 Eigen::Vector3d omg;
                 self.forward(vel, acc, jer, psi, dpsi, thrust, quat, omg);
                 return py::make_tuple(thrust, quat, omg);
             },
             py::arg("vel"), py::arg("acc"), py::arg("jer"), py::arg("psi"), py::arg("dpsi"),
             "Run the forward flatness map and return (thrust, quaternion, body_rates).")
        .def("backward",
             [](const flatness::FlatnessMap &self,
                const Eigen::Vector3d &pos_grad,
                const Eigen::Vector3d &vel_grad,
                const double &thr_grad,
                const Eigen::Vector4d &quat_grad,
                const Eigen::Vector3d &omg_grad) {
                 Eigen::Vector3d pos_total_grad;
                 Eigen::Vector3d vel_total_grad;
                 Eigen::Vector3d acc_total_grad;
                 Eigen::Vector3d jer_total_grad;
                 double psi_total_grad = 0.0;
                 double dpsi_total_grad = 0.0;
                 self.backward(pos_grad, vel_grad, thr_grad, quat_grad, omg_grad,
                               pos_total_grad, vel_total_grad, acc_total_grad,
                               jer_total_grad, psi_total_grad, dpsi_total_grad);
                 return py::make_tuple(pos_total_grad, vel_total_grad, acc_total_grad,
                                       jer_total_grad, psi_total_grad, dpsi_total_grad);
             },
             py::arg("pos_grad"), py::arg("vel_grad"), py::arg("thr_grad"),
             py::arg("quat_grad"), py::arg("omg_grad"),
             "Run the adjoint flatness map and return gradients with respect to flat outputs.");
}
