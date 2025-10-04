# Repository Guidelines

## Project Structure & Module Organization
- Core bindings live under `src/minco_trajectory/`, split into `include/` (header-only math + planners), `src/` (pybind11 entrypoints), and `config/` (YAML tuning presets). 
- The Python extension builds to `minco.*.so` plus stubs at the repo root, so downstream notebooks can simply `import minco`.
- Tests reside in `tests/`, organized by binding surface (`test_polynomial_bindings.py`, `test_gcopter_bindings.py`, etc.). Automation scripts belong in `scripts/`.

## Build, Test, and Development Commands
- `uv sync` — install Python dependencies and dev tooling into the uv-managed virtual env.
- `uv pip install -e . --no-deps` — compile the pybind11 module in editable mode; rerun after any C++ changes.
- `uv run pytest` — execute the full test suite; scope with `-k` for targeted checks.
- `uv run ruff check .` / `uv run ruff format .` — lint and auto-format Python surfaces.

## Coding Style & Naming Conventions
- Python Type Hinting All the time
- Python follows PEP 8 with four-space indents and snake_case modules; classes use UpperCamelCase.
- C++ headers stay header-only when feasible; favor descriptive Eigen typedefs and keep namespaces aligned with file paths.
- Match config keys in YAML (`config/global_planning.yaml`) with snake_case accessors in code.
- Prefer declarative, NumPy-friendly APIs; avoid imperative helpers unless profiling demands it.

## Testing Guidelines
- Pytest discovers `tests/test_*.py` with classes prefixed `Test`. Target 80%+ coverage on modified logic.
- Leverage fixtures and numerical tolerances (e.g., `pytest.approx`, `np.testing.assert_allclose`) for dynamics or gradient checks.
- Use WebAgg when plotting (`matplotlib.use("WebAgg")`) to keep CI headless-friendly.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative subjects (e.g., "Expose gcopter bindings"). Include rationale or TODOs in bodies only when behavior changes.
- PRs should link relevant issues, summarize interface impacts, and list command logs (`uv run pytest`, `uv run ruff check .`). Attach plots or screenshots for planner visualizations and flag C++ ABI changes so reviewers rebuild locally.

## Agent-Specific Tips
- Regenerate stubs after altering signatures: `uv run pybind11-stubgen minco && mv stubs/minco.pyi .`.
- When modifying GCOPTER penalties, update both the C++ flatness integration and its Python smoke tests (`tests/test_gcopter_bindings.py`).
