from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import subprocess
import sys

EIGEN_INCLUDE_DIR = "/usr/include/eigen3"
PROJETCT_INCLUDE_DIR = "src/minco_trajectory/include"

ext_modules = [
    Extension(
        "minco",
        [
            "src/minco_trajectory/src/minco_module.cpp",
            "src/minco_trajectory/src/bindings/polynomial_bindings.cpp",
            "src/minco_trajectory/src/bindings/sdlp_bindings.cpp",
            "src/minco_trajectory/src/bindings/root_finder_bindings.cpp",
            "src/minco_trajectory/src/bindings/geo_utils_bindings.cpp",
            "src/minco_trajectory/src/bindings/flatness_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), PROJETCT_INCLUDE_DIR, EIGEN_INCLUDE_DIR],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2"],
    ),
]


# 自定义 build 命令：生成类型提示
class BuildWithStubs(build_ext):
    def run(self):
        # 1. 先编译 C++ 模块
        super().run()

        # 2. 生成类型提示（.pyi）
        try:
            # 使用 subprocess 调用 pybind11-stubgen
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pybind11_stubgen",
                    "--module-name=minco",
                    "--output-dir=./stubs",
                    "--ignore-invalid=all",
                ],
                check=True,
            )

            os.system("cp ./stubs/minco.pyi ./")
        except Exception as e:
            print(f"Warning: Failed to generate stubs - {e}")


setup(
    name="minco-python",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildWithStubs},
    packages=[],
    package_data={},
    install_requires=["pybind11>=2.10.0", "pybind11-stubgen>=0.12"],
    python_requires=">=3.7",
)
