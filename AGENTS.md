# Agent Playbook

## 1. Repository Overview

- Core C++ bindings live under `_src/minco_trajectory/` and split into:
  - `include/`: header-only math utilities and trajectory planners.
  - `src/`: pybind11 binding entrypoints.
  - `config/`: YAML presets for flatness, cost, and planner tuning.
- The C++ extension builds to `_minco.*.so` via `CMakeLists.txt` + scikit-build-core.
- `minco/` is the public Python package: `__init__.py` re-exports from `_minco`.
- `minco/flatness_cache.py` provides B2 on-demand CasADi flatness with disk caching.
- Tests live in `tests/`. Examples live in `examples/`.
- Generated artifacts (plots, logs, etc.) land in `_tmp/` (gitignored).

## 2. Everyday Commands

- `uv sync` — build C++ extension via CMake + scikit-build-core and install dependencies.
- `uv run pytest` — execute the full Python test suite (`-k` to scope).
- `uv run ruff check .` and `uv run ruff format .` — lint and format Python code.
- `uv run demo flatness` — demonstrate flatness forward/backward.
- `uv run demo trajectory` — demonstrate waypoint-to-trajectory pipeline (BEV + kinematic profiles).
- `python -m minco.flatness_cache --list` / `--clear` — manage cached flatness libraries.
- `python -m minco.flatness_cache --generate-c --config config/casadi_quadrotor_flatness.yaml` — regenerate embedded C sources.

## 3. Coding Standards

### Python

- Always add type hints to public functions, tests, and helpers.
- Follow PEP 8: four-space indentation, snake_case modules and functions, UpperCamelCase classes.
- Prefer declarative, NumPy-friendly code over imperative loops unless profiling justifies otherwise.

### C++

- Keep headers header-only when feasible and align namespaces with directory paths.
- Use descriptive Eigen typedefs and mirror YAML keys with snake_case accessors in code.

## 4. Testing Guidelines

- Pytest discovers `tests/test_*.py`; target ≥80% coverage on modified logic.
- Use fixtures and numerical tolerances (`pytest.approx`, `np.testing.assert_allclose`) for dynamics/gradient checks.
- Force WebAgg in plotting tests (`matplotlib.use("WebAgg")`) to stay headless-friendly.

## 5. Git & Review Workflow

- Commit messages use short, imperative subjects (e.g. `Add casadi regression test`).
- Pull requests should link issues, call out interface changes, and include command logs (`uv run pytest`, `uv run ruff check .`). Attach plots or screenshots when visual behaviour changes and flag any C++ ABI impact.

## 6. Agent-Specific Practices

- Regenerate Python stubs after C++ signature changes:
  ```bash
  uv run python -m pybind11_stubgen minco._minco --output-dir ./stubs --ignore-all-errors
  ```
- Extension module is `_minco` (private). Public API is `minco/__init__.py` + `minco.pyi` stubs.
- Build system is `CMakeLists.txt` + scikit-build-core (no `setup.py`).
- Prefer adding new configuration through YAML files in `config/` rather than hardcoding constants.
- When tweaking GCOPTER penalties, update the matching C++ integration and the smoke tests in `tests/test_gcopter_bindings.py`.
- For B2 flatness caching: generate C sources with `python -m minco.flatness_cache --generate-c`, rebuild with `uv sync`.
