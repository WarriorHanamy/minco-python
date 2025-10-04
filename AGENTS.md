# Agent Playbook

## 1. Repository Overview
- Core bindings live under `src/minco_trajectory/` and split into:
  - `include/`: header-only math utilities and trajectory planners.
  - `src/`: pybind11 binding entrypoints.
  - `config/`: YAML presets for flatness, cost, and planner tuning.
- The Python extension builds to `minco.*.so` plus interface stubs at the repo root so downstream notebooks can `import minco` directly.
- Tests live in `tests/` and mirror the exposed binding surfaces (e.g. `test_gcopter_bindings.py`, `test_flatness_bindings.py`). Support scripts belong in `scripts/`.

## 2. Everyday Commands
- `uv sync` — install dependencies into the uv-managed virtual environment.
- `uv pip install -e . --no-deps` — rebuild the pybind11 extension after C++ changes.
- `uv run pytest` — execute the full Python test suite (`-k` to scope).
- `uv run ruff check .` and `uv run ruff format .` — lint and format Python code.

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
- Regenerate Python stubs after signature changes with `uv run pybind11-stubgen minco && mv stubs/minco.pyi .`.
- When tweaking GCOPTER penalties, update the matching C++ integration and the smoke tests in `tests/test_gcopter_bindings.py`.
- Prefer adding new configuration through YAML files in `config/` rather than hardcoding constants.

## 7. Debug Artifact Policy
- Place all generated debugging artifacts under `codex/debug/<behavior>/`.
- After completing a debugging session, add a markdown report `codex/debug/<behavior>/<behavior>.md` summarizing:
  1. The debugging activity performed.
  2. The motivation or issue being investigated.
  3. The outcome, including follow-up actions if needed.
- Remove temporary artifacts once the report is committed to keep the workspace clean.
