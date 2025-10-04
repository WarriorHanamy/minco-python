#include <pybind11/pybind11.h>

#include "bindings/bindings.hpp"

namespace py = pybind11;

// Entry point that exposes the MINCO bindings under the `minco` module.
PYBIND11_MODULE(minco, m)
{
    m.doc() = "Bindings for MINCO trajectory planning";

    auto poly_traj = m.def_submodule("poly_traj", "Polynomial poly_traj primitives");
    bind_polynomial(poly_traj);

    bind_sdlp(m);
    bind_root_finder(m);
    bind_geo_utils(m);
    bind_flatness(m);
    bind_gcopter(m);
}
