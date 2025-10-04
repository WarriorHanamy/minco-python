#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "bindings.hpp"
#include "gcopter.hpp"

namespace py = pybind11;

void bind_gcopter(py::module_ &m)
{
    auto sub = m.def_submodule("gcopter", "Trajectory optimization helpers built on GCOPTER");

    py::class_<gcopter::GCOPTER_PolytopeSFC>(sub, "GCOPTERPolytopeSFC")
        .def(py::init<>())
        .def("configure_flatness",
             [](gcopter::GCOPTER_PolytopeSFC &self, double mass, double gravity,
                double horizontal_drag, double vertical_drag,
                double parasitic_drag, double speed_smooth, double yaw_smooth) {
                 self.configure_flatness(mass, gravity, horizontal_drag,
                                         vertical_drag, parasitic_drag,
                                         speed_smooth, yaw_smooth);
             },
             py::arg("mass") = 1.0, py::arg("gravity") = 9.81,
             py::arg("horizontal_drag") = 0.0, py::arg("vertical_drag") = 0.0,
             py::arg("parasitic_drag") = 0.0, py::arg("speed_smooth") = 1.0e-3,
             py::arg("yaw_smooth") = 1.0e-6)
        .def("configure_cost",
             [](gcopter::GCOPTER_PolytopeSFC &, const py::kwargs &kwargs) {
                 auto &cfg = gcopter::CostConfig::getInstance();
                 for (auto &item : kwargs)
                 {
                     const std::string key = py::str(item.first);
                     const double      value = py::cast<double>(item.second);

                     if (key == "v_max")
                     {
                         cfg.v_max = value;
                     }
                     else if (key == "omg_x_max")
                     {
                         cfg.omg_x_max = value;
                     }
                     else if (key == "omg_y_max")
                     {
                         cfg.omg_y_max = value;
                     }
                     else if (key == "omg_z_max")
                     {
                         cfg.omg_z_max = value;
                     }
                     else if (key == "acc_max")
                     {
                         cfg.acc_max = value;
                     }
                     else if (key == "thrust_min")
                     {
                         cfg.thrust_min = value;
                     }
                     else if (key == "thrust_max")
                     {
                         cfg.thrust_max = value;
                     }
                     else if (key == "pos_weight")
                     {
                         cfg.pos_weight = value;
                     }
                     else if (key == "vel_weight")
                     {
                         cfg.vel_weight = value;
                     }
                     else if (key == "acc_weight")
                     {
                         cfg.acc_weight = value;
                     }
                     else if (key == "omg_x_weight")
                     {
                         cfg.omg_x_weight = value;
                     }
                     else if (key == "omg_y_weight")
                     {
                         cfg.omg_y_weight = value;
                     }
                     else if (key == "omg_z_weight")
                     {
                         cfg.omg_z_weight = value;
                     }
                     else if (key == "thrust_weight")
                     {
                         cfg.thrust_weight = value;
                     }
                     else if (key == "time_weight")
                     {
                         cfg.time_weight = value;
                     }
                     else if (key == "omg_consistent_weight")
                     {
                         cfg.omg_consistent_weight = value;
                     }
                     else
                     {
                         throw py::key_error("Unsupported cost config key: " +
                                              key);
                     }
                 }
             },
             "Update GCOPTER cost configuration. Accepts keyword arguments.")
        .def("setup_basic_trajectory",
             &gcopter::GCOPTER_PolytopeSFC::setup_basic_trajectory,
             py::arg("initial_pva"), py::arg("terminal_pva"),
             py::arg("initial_time"), py::arg("initial_points"),
             py::arg("sfc_control_points"), py::arg("smoothing_factor"),
             py::arg("integral_resolution"))
        .def("optimize",
             [](gcopter::GCOPTER_PolytopeSFC &self, double rel_cost_tol) {
                 Trajectory<5> traj;
                 double       cost = self.optimize(traj, rel_cost_tol);
                 return py::make_tuple(cost, traj);
             },
             py::arg("rel_cost_tol"));
}
