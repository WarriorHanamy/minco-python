# MINCO-Python Project Code Style Guide

## Overview
This project is a Python binding for MINCO (Minimum Control Effort) trajectory optimization library, providing efficient trajectory planning capabilities.

## Code Style Conventions

### Python Code Style
- **Imports**: Standard library imports first, then third-party, then local imports
- **Function naming**: snake_case for functions and variables
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Line length**: ~80-100 characters
- **Visualization**: Uses matplotlib with WebAgg backend for interactive plots
- **Testing**: Uses pytest-style assertions and numpy for numerical operations

### C++ Code Style
- **Header guards**: `#pragma once` for header files
- **Namespace**: `minco` namespace for core functionality
- **Naming**: PascalCase for classes, camelCase for methods and variables
- **Inline functions**: Heavy use of `inline` for performance-critical code
- **Eigen integration**: Extensive use of Eigen library for linear algebra
- **Memory management**: Manual memory management with `new`/`delete` for performance
- **Template usage**: Template metaprogramming for generic algorithms

### File Organization
- **Python**: `tests/` directory for test files, `setup.py` for build configuration
- **C++**: 
  - Headers in `src/minco_trajectory/include/`
  - Source in `src/minco_trajectory/src/`
  - Bindings in `src/minco_trajectory/src/bindings/`
- **Build system**: Uses setuptools with pybind11 for Python bindings

### Key Patterns
- **Trajectory optimization**: MINCO_S2NU, MINCO_S3NU, MINCO_S4NU classes for different smoothness orders
- **Banded systems**: Efficient banded matrix solvers for trajectory constraints
- **Polynomial trajectories**: Piecewise polynomial representation
- **GCOPTER integration**: Geometric control and trajectory optimization

### Testing Conventions
- Test functions start with `test_` prefix
- Uses numpy arrays for numerical data
- Matplotlib for visualization in tests
- Assertions for validation of numerical results

### Build Configuration
- C++17 standard with optimization flags (`-O2`)
- Eigen3 and pybind11 dependencies
- Custom build commands for stub generation