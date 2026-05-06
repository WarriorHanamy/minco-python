cmake_minimum_required(VERSION 3.15...3.31)
project(minco-python VERSION 0.1.0 LANGUAGES CXX C)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

pybind11_add_module(_minco
    _src/minco_trajectory/src/minco_module.cpp
    _src/minco_trajectory/src/bindings/polynomial_bindings.cpp
    _src/minco_trajectory/src/bindings/sdlp_bindings.cpp
    _src/minco_trajectory/src/bindings/root_finder_bindings.cpp
    _src/minco_trajectory/src/bindings/geo_utils_bindings.cpp
    _src/minco_trajectory/src/bindings/flatness_bindings.cpp
    _src/minco_trajectory/src/bindings/gcopter_bindings.cpp
    _src/minco_trajectory/src/casadi_generated/quadrotor_flatness.c
)

target_include_directories(_minco PRIVATE
    _src/minco_trajectory/include
)

target_compile_features(_minco PRIVATE cxx_std_20)
target_compile_options(_minco PRIVATE -O2)

target_link_libraries(_minco PRIVATE
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
)

install(TARGETS _minco LIBRARY DESTINATION minco)
