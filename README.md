# MINCO-Python

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Python bindings for MINCO (Minimum Control Effort) trajectory optimization library** - A high-performance trajectory planning framework for multirotor and fixed-wing aircraft.

![MINCO Trajectory — line](./thumbnail_line.png)
![MINCO Trajectory — circle](./thumbnail_circle.png)
![MINCO Trajectory — fig8](./thumbnail_fig8.png)

## Overview

MINCO-Python is a Python-first trajectory optimization library that provides efficient trajectory planning capabilities for unmanned aerial vehicles. Based on the [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER.git) framework, this project removes ROS dependencies and exposes a clean Python interface while maintaining the high-performance C++ backend.

## Key Features

- 🚀 **Python-First Design**: Native Python interface with NumPy integration
- ⚡ **High Performance**: C++20 backend with O(N) banded system solvers
- 🎯 **CasADi Integration**: Automatic differentiation for flatness models
- 🔧 **Configurable**: YAML-based configuration system
- 📊 **Visualization**: Built-in matplotlib support with interactive plotting
- 🧪 **Well-Tested**: Comprehensive test suite with validation examples

## Architecture

### Core Components

**C++ Backend:**
- `minco.hpp` - Banded system solver with O(N) complexity
- `gcopter.hpp` - GCOPTER trajectory optimizer with geometric control
- `flatness.hpp` - Differential flatness mapping for multirotor dynamics
- `trajectory.hpp` - Piecewise polynomial trajectory representation

**Python Bindings:**
- Complete pybind11 bindings with NumPy array support
- Automatic Python type stub generation (.pyi files)
- CasADi automatic differentiation integration

**Configuration System:**
- YAML configuration files for aircraft parameters
- Customizable flatness model definitions
- LBFGS optimizer parameter tuning

## Installation

### Prerequisites

- Python 3.12+
- C++20 compatible compiler (gcc or clang)
- CMake 3.15+
- Eigen3 library
- yaml-cpp library

### Install C++ Dependencies

```bash
# Install required C++ libraries
sudo apt install libyaml-cpp-dev libeigen-dev
```

### Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Python Dependencies

```bash
# Install all Python dependencies
uv sync
```
It would automatically building the cpp project.
<!-- 
```bash
# Build C++ extensions and install Python package
uv pip install -e . --no-deps
``` -->

### Generate Type Stubs [Optional]

```bash
# Generate pybind11 type hints for better IDE support
uv run python -m pybind11_stubgen minco._minco --output-dir ./stubs --ignore-all-errors
```

The generated stub files under `stubs/minco/_minco/` are re-exported by the
committed `minco.pyi` file at the repo root.

## Quick Start

```python
import numpy as np
import minco

# 1. Configure optimizer
opt = minco.gcopter.GCOPTERPolytopeSFC()
opt.configure_from_file("")  # uses default config

# 2. Define problem: initial/final position-velocity-acceleration
head_pva = np.column_stack([np.zeros(3), np.zeros(3), np.zeros(3)])
tail_pva = np.column_stack([np.array([5.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)])

# Waypoints (3 x N)
inner_points = np.array([[2.5], [0.0], [0.5]])
initial_time = np.array([2.0, 2.0])

# Safe flight corridors: each is (M, 4) half-space matrix
sfc = [np.array([
    [1, 0, 0, -1], [-1, 0, 0, -1],
    [0, 1, 0, -1], [0, -1, 0, -1],
    [0, 0, 1, -1], [0, 0, -1, -1],
], dtype=float)]

# 3. Setup and optimize
opt.setup_basic_trajectory(head_pva, tail_pva, initial_time,
                           inner_points, sfc,
                           smoothing_factor=1e-1, integral_resolution=24)
cost, traj = opt.optimize(rel_cost_tol=1e-3)

# 4. Evaluate trajectory
print(f"Cost: {cost:.2f}, Duration: {traj.total_duration:.2f}s, Pieces: {traj.get_piece_num()}")
for t in np.linspace(0, traj.total_duration, 5):
    print(f"  t={t:.1f}s  pos={traj.get_pos(t)}")
```

Run the full demo:
```bash
uv run demo trajectory          # all shapes: line, circle, fig8
uv run demo trajectory line     # line_traj only
uv run demo flatness            # flatness forward/backward
```

## Flatness Caching (B2)

The `minco.flatness_cache` module provides on-demand CasADi flatness compilation
with automatic disk caching.  Parameters are hashed with SHA256; the first call
compiles the flatness model to a shared library via CasADi + gcc, and subsequent
calls load from `~/.cache/minco/flatness/<hash>.so` instantly.

```python
from minco.flatness_cache import CachedFlatness

cf = CachedFlatness(mass=1.0, gravity=9.81, horizontal_drag=0.1,
                    vertical_drag=0.1, parasitic_drag=0.01, speed_smooth=1e-3)

thrust, quat, omg = cf.forward(vel, acc, jer, yaw, yaw_rate)
pg, vg, ag, jg, psig, dpsig = cf.backward(pos_grad, vel_grad, thr_grad,
                                           quat_grad, omg_grad)
```

Cache management:
```bash
python -m minco.flatness_cache --list    # list cached parameter sets
python -m minco.flatness_cache --clear   # clear all cached .so files
python -m minco.flatness_cache --info    # show cache directory and disk usage
```

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_gcopter_casadi_visualization.py
```

### Test Categories

- **Flatness Tests**: Validate differential flatness implementations
- **GCOPTER Tests**: Test trajectory optimization with various constraints
- **Visualization Tests**: Interactive trajectory plotting and validation
- **CasADi Tests**: Automatic differentiation and gradient computation



## Development Roadmap

- [ ] **LBFGS → SQCQP**: Upgrade optimization algorithm
- [ ] **Enhanced API**: More natural Python interface design
- [ ] **Fixed-Wing Support**: Add fixed-wing/VTOL differential flatness models
- [ ] **Real-time Planning**: Real-time trajectory generation capabilities

## Configuration

Configuration files are located in the `config/` directory:

- `default_gcopter.yaml` - Main trajectory optimization parameters
- `default_flatness_config.yaml` - Flatness model parameters
- `lbfgs.yaml` - Optimizer configuration
- `casadi_quadrotor_flatness.yaml` - CasADi flatness model

### Key Configuration Sections

1. **Cost Function Configuration** (`costfunc_config`)
   - Weight matrices for position, velocity, acceleration
   - Physical constraints (velocity, thrust limits)
   - Smoothness factors

2. **LBFGS Optimizer Configuration** (`lbfgs_config`)
   - Convergence tolerances
   - Maximum iterations
   - Line search parameters

## Supported Trajectory Shapes

The library includes built-in trajectory generators for:

- 🔵 **Circular** trajectories
- **8-Shaped** trajectories
- **Square** trajectories
- Custom waypoint-based trajectories


## Applications

- Multirotor UAV trajectory planning
- Learning-based control research
- Aircraft dynamics simulation
- Trajectory optimization algorithm validation
- Real-time motion planning

## Project Structure

```
minco-python/
├── CMakeLists.txt                # C++ build configuration (scikit-build-core)
├── _src/                         # Internal C++ pybind11 sources
│   └── minco_trajectory/
│       ├── include/              # Header-only math/trajectory/planner code
│       ├── src/                  # Implementation & bindings
│       └── config/               # Embedded YAML presets
├── minco/                        # Public Python package
│   ├── __init__.py               # Re-exports from compiled _minco extension
│   ├── flatness_cache.py         # B2 on-demand CasADi flatness with caching
│   └── py.typed                  # PEP 561 marker
├── examples/                     # User-facing demo scripts
│   ├── demo_flatness.py
│   └── demo_trajectory.py
├── tests/                        # Test suite
├── config/                       # Runtime configuration files
├── _tmp/                         # Generated artifacts (gitignored)
└── pyproject.toml                # Package metadata & build config
```

## Technical Details

### Differential Flatness

The project uses differential flatness theory to transform complex 3D trajectory planning problems into simpler flat space optimization. The flatness mapping handles:

- Forward mapping: flat variables → physical states (position, velocity, acceleration)
- Backward mapping: physical state gradients → flat variable gradients

### Optimization

Trajectory optimization is performed using LBFGS with:
- Piecewise polynomial trajectory representation
- Physical constraints (velocity, acceleration, thrust limits)
- Smoothness regularization terms
- Boundary condition enforcement

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) framework from ZJU FAST Lab
- Built with [Eigen](https://eigen.tuxfamily.org) for linear algebra
- Python bindings powered by [pybind11](https://github.com/pybind/pybind11)
- Symbolic computation with [CasADi](https://web.casadi.org)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{minco_python,
  title={MINCO-Python: Python Bindings for Minimum Control Effort Trajectory Optimization},
  author={Erchao Rong},
  year={2025},
  url={https://github.com/WarriorHanamy/minco-python}
}
```
