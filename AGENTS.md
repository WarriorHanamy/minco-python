# Repository Guidelines

## Project Structure & Module Organization
The python-first interface lives in `src/minco_trajectory`, which is split into `src/` for pybind11 glue code, `include/` for C++ headers, and `config/` for tuning YAML files. The compiled extension and stubs (`trajectory.*.so`, `trajectory.pyi`) are kept at the project root so downstream notebooks can import `trajectory` directly. Python-facing tests reside in `tests/`, and auxiliary automation belongs in `scripts/` (add new task runners here when needed).

## Build, Test, and Development Commands
Run `uv sync` to pull Python dependencies and dev tools. Build the native extension with `uv pip install -e . --no-deps`; this compiles the pybind11 module and links Eigen. Use `uv run pytest` for the functional suite, and `uv run pytest -k name` to focus on a scenario. Lint and format via `uv run ruff check .` and `uv run ruff format .`. When C++ surfaces change, refresh stubs with `uv run pybind11-stubgen trajectory && mv stubs/trajectory.pyi .`.

When plotting, always choose "matplotlib.use("WebAgg")  " by default.

## Coding Style & Naming Conventions
Favor declarative, data-first APIs that compose cleanly with NumPy and CasADi pipelines; resist imperative helpers unless performance-bound. Python code follows PEP 8 with four-space indents, snake_case modules, and UpperCamelCase classes. Mirror configuration files on disk with matching snake_case identifiers in code. Keep docstrings concise and type hint public call surfaces. C++ headers should stay header-only when possible to preserve the declarative-first approach.

## Testing Guidelines
`pytest` discovers files named `test_*.py` and classes prefixed `Test`. Add regression cases under `tests/` that cover both Python orchestrators and pybind11 bridge behavior; prefer fixture-driven inputs over print-based assertions. Target meaningful coverage for new logic (aim for 80%+ of touched lines) and verify gradients or dynamics with numerical tolerances rather than exact matches.

## Commit & Pull Request Guidelines
Project history favors short, imperative commit subjects (e.g., "Remove junky things"). Keep bodies minimal but include rationale or follow-up TODOs when behavior changes. Pull requests should link issues, describe interface impacts, list test runs (`uv run pytest`, `uv run ruff check .`), and attach logs or screenshots for planner visualizations. Flag any C++ ABI changes so reviewers can rebuild locally.

## Stub & Native Module Tips
When editing C++ bindings, sync signatures between `traj_bindings.cpp` and `trajectory.pyi`. Validate Eigen include paths remain correct before pushing. Document new configuration options in `config/*.yaml` and surface them through Python wrappers so downstream agents stay python-first.
