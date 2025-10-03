#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "trajectory.hpp" // Your provided header file

namespace py = pybind11;

// Template helper function to bind Piece and Trajectory for a specific degree D
template <int D>
void bind_trajectory_degree(py::module &m)
{
    std::string suffix = std::to_string(D);
    using PieceD = Piece<D>;
    using TrajectoryD = Trajectory<D>;
    using CoeffMatD = typename PieceD::CoefficientMat;

    // Bind the Piece class
    py::class_<PieceD>(m, ("Piece" + suffix).c_str())
        .def(py::init<>(), "Default constructor.")
        .def(py::init<double, const CoeffMatD &>(), py::arg("duration"), py::arg("coeff_mat"),
             "Construct a piece with a duration and a coefficient matrix.")
        .def_property_readonly("duration", &PieceD::getDuration, "Get the duration of the piece.")
        .def_property_readonly("degree", &PieceD::getDegree, "Get the polynomial degree of the piece.")
        .def_property_readonly("dim", &PieceD::getDim, "Get the spatial dimension (always 3).")
        .def("get_coeff_mat", &PieceD::getCoeffMat, py::return_value_policy::copy,
             "Get the coefficient matrix (3x(D+1) NumPy array).")
        .def("get_pos", &PieceD::getPos, py::arg("t"), "Get position at time t.")
        .def("get_vel", &PieceD::getVel, py::arg("t"), "Get velocity at time t.")
        .def("get_acc", &PieceD::getAcc, py::arg("t"), "Get acceleration at time t.")
        .def("get_jer", &PieceD::getJer, py::arg("t"), "Get jerk at time t.")
        .def("get_max_vel_rate", &PieceD::getMaxVelRate, "Get the maximum velocity magnitude in this piece.")
        .def("get_max_acc_rate", &PieceD::getMaxAccRate, "Get the maximum acceleration magnitude in this piece.")
        .def("check_max_vel_rate", &PieceD::checkMaxVelRate, py::arg("max_vel_rate"),
             "Check if the velocity magnitude is always within a certain limit.")
        .def("check_max_acc_rate", &PieceD::checkMaxAccRate, py::arg("max_acc_rate"),
             "Check if the acceleration magnitude is always within a certain limit.")
        .def("__repr__",
             [suffix](const PieceD &p) {  // Capture suffix by value
                 return "<Piece" + suffix + " duration=" + std::to_string(p.getDuration()) + ">";
             });

    // Bind the Trajectory class
    py::class_<TrajectoryD>(m, ("Trajectory" + suffix).c_str())
        .def(py::init<>(), "Default constructor.")
        .def(py::init<const std::vector<double> &, const std::vector<CoeffMatD> &>(),
             py::arg("durations"), py::arg("coeff_mats"),
             "Construct a trajectory from a list of durations and a list of coefficient matrices.")
        .def_property_readonly("total_duration", &TrajectoryD::getTotalDuration, "Get the total duration of the trajectory.")
        .def_property_readonly("durations", &TrajectoryD::getDurations, "Get a NumPy array of all piece durations.")
        .def_property_readonly("positions", &TrajectoryD::getPositions, "Get a NumPy array of all waypoint positions.")
        .def("get_piece_num", &TrajectoryD::getPieceNum, "Get the number of pieces in the trajectory.")
        .def("get_pos", &TrajectoryD::getPos, py::arg("t"), "Get position at time t along the trajectory.")
        .def("get_vel", &TrajectoryD::getVel, py::arg("t"), "Get velocity at time t along the trajectory.")
        .def("get_acc", &TrajectoryD::getAcc, py::arg("t"), "Get acceleration at time t along the trajectory.")
        .def("get_jer", &TrajectoryD::getJer, py::arg("t"), "Get jerk at time t along the trajectory.")
        .def("clear", &TrajectoryD::clear, "Clear all pieces from the trajectory.")
        .def("append_piece", static_cast<void (TrajectoryD::*)(const PieceD &)>(&TrajectoryD::emplace_back),
             py::arg("piece"), "Append a piece object to the trajectory.")
        .def("get_max_vel_rate", &TrajectoryD::getMaxVelRate, "Get the maximum velocity magnitude along the entire trajectory.")
        .def("get_max_acc_rate", &TrajectoryD::getMaxAccRate, "Get the maximum acceleration magnitude along the entire trajectory.")
        
        // Pythonic protocols
        .def("__len__", &TrajectoryD::getPieceNum)
        .def("__getitem__", [](const TrajectoryD &traj, int i) {
            if (i < 0) i += traj.getPieceNum();
            if (i < 0 || i >= traj.getPieceNum()) throw py::index_error();
            return traj[i];
        }, py::return_value_policy::reference_internal)
        .def("__iter__", [](const TrajectoryD &traj) {
            return py::make_iterator(traj.begin(), traj.end());
        }, py::keep_alive<0, 1>()) // Keep trajectory object alive while iterator is used
        .def("__repr__",
             [suffix](const TrajectoryD &t) {  // Capture suffix by value
                 return "<Trajectory" + suffix + " with " + std::to_string(t.getPieceNum()) +
                        " pieces, total_duration=" + std::to_string(t.getTotalDuration()) + ">";
             });
}


PYBIND11_MODULE(trajectory, m)
{
    m.doc() = "Pybind11 bindings for the polynomial trajectory representation";

    // Create bindings for specific polynomial degrees
    bind_trajectory_degree<5>(m);
    bind_trajectory_degree<7>(m);
}