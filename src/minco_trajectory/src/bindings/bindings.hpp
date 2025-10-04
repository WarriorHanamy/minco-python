#pragma once

#include <pybind11/pybind11.h>

void bind_polynomial(pybind11::module_ &m);
void bind_sdlp(pybind11::module_ &m);
void bind_root_finder(pybind11::module_ &m);
void bind_geo_utils(pybind11::module_ &m);
void bind_flatness(pybind11::module_ &m);
void bind_gcopter(pybind11::module_ &m);
