import setuptools

from pybind11.setup_helpers import Pybind11Extension  # type: ignore
import pybind11  # type: ignore

EIGEN_INCLUDE_DIR = "/usr/include/eigen3"
PROJETCT_INCLUDE_DIR = "src/minco_trajectory/include"

ext_modules = [
    Pybind11Extension(
        "trajectory",
        ["src/minco_trajectory/src/traj_bindings.cpp"],
        include_dirs=[pybind11.get_include(), PROJETCT_INCLUDE_DIR, EIGEN_INCLUDE_DIR],
        cxx_std=17,
    ),
]

setuptools.setup(
    name="trajectory",
    ext_modules=ext_modules,
    version="0.1.0",
    author="Your Name",
    description="A pybind11 for polynomial trajectory representation",
)
