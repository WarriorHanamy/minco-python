#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "bindings.hpp"
#include "flatness.hpp"
#include "flatness_casadi.hpp"

namespace py = pybind11;

void bind_flatness(py::module_ &m)
{
    auto sub = m.def_submodule("flatness", "Forward and backward flatness transforms");
    py::class_<minco::flatness::FlatnessMap>(sub, "FlatnessMap")
        .def(py::init<>())
        .def("configure_from_file",
             &minco::flatness::FlatnessMap::configure_from_file,
             py::arg("file_path") = std::string(),
             "Load flatness parameters from a YAML configuration file (empty path selects the default bundle).")
        .def("forward",
             [](minco::flatness::FlatnessMap &self,
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
             [](const minco::flatness::FlatnessMap &self,
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

    py::class_<minco::flatness::CasadiQuadrotorFlatnessMap>(sub, "CasadiQuadrotorFlatnessMap")
        .def(py::init<>())
        .def("configure_from_file",
             &minco::flatness::CasadiQuadrotorFlatnessMap::configure_from_file,
             py::arg("file_path") = std::string(),
             "CasADi flatness maps are generated ahead-of-time; pass an empty path or regenerate code with new parameters.")
        .def("forward",
             [](minco::flatness::CasadiQuadrotorFlatnessMap &self,
                const Eigen::Vector3d &vel,
                const Eigen::Vector3d &acc,
                const Eigen::Vector3d &jer,
                double psi,
                double dpsi) {
                 const auto result = self.forward({vel, acc, jer, psi, dpsi});
                 return py::make_tuple(result.thrust, result.quaternion, result.angular_velocity);
             },
             py::arg("vel"), py::arg("acc"), py::arg("jer"), py::arg("psi"), py::arg("dpsi"),
             "Run the CasADi flatness map and return (thrust, quaternion, body_rates).")
        .def("backward",
             [](minco::flatness::CasadiQuadrotorFlatnessMap &self,
                const Eigen::Vector3d &pos_grad,
                const Eigen::Vector3d &vel_grad,
                const double &thr_grad,
                const Eigen::Vector4d &quat_grad,
                const Eigen::Vector3d &omg_grad) {
                 const auto result = self.backward({pos_grad, vel_grad, thr_grad, quat_grad, omg_grad});
                 return py::make_tuple(result.position_total_gradient,
                                       result.velocity_total_gradient,
                                       result.acceleration_total_gradient,
                                       result.jerk_total_gradient,
                                       result.yaw_total_gradient,
                                       result.yaw_rate_total_gradient);
             },
             py::arg("pos_grad"), py::arg("vel_grad"), py::arg("thr_grad"),
             py::arg("quat_grad"), py::arg("omg_grad"),
             "Run the CasADi adjoint map and return gradients with respect to flat outputs.");
}
