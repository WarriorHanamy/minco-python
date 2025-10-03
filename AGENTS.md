# Repository Guidelines

## Coding Guidelines
- python-first
- declartive-first 

## Project Structure & Module Organization
- `src/trajectory_server` is the ROS 2 package driving GCOPTER-based global planning. The `include/gcopter` headers mirror the original solver, while `src/global_planning.cpp` hosts the ROS node wiring trajectory generation.
- Runtime parameters live under `src/trajectory_server/config/` (YAML, RViz) and fleet-level defaults in `global.toml`. Update both when adding new knobs.
- Future Python-first utilities belong under the `vtol_rl` package declared in `pyproject.toml`; co-locate tests in `vtol_rl/tests`.

## Build, Test, and Development Commands
- `source pnc_setup.bash` to load the expected ROS Humble, PX4, and acados paths before building.
- `colcon build --packages-select trajectory_server` compiles the C++ node with the configured ament dependencies.
- `colcon test --packages-select trajectory_server` runs ROS 2 tests once you add them.
- `pytest` (default target `vtol_rl/tests`) is available for Python modules; use `pytest --maxfail=1 --disable-warnings` when iterating.
- `pre-commit run --all-files` applies Ruff linting/formatting and repo hygiene checks.

## Coding Style & Naming Conventions
- C++ follows `.clang-format` (Google base, 4-space Allman braces); run `clang-format` or rely on `pre-commit`. Prefer PascalCase for types, lowerCamelCase for members to match `GlobalPlanner`.
- Keep ROS parameters snake_case to align with existing YAML keys.
- Python code should pass Ruff linting/formatting; favour PEP 8 names and type hints.

## Testing Guidelines
- Add deterministic unit coverage for planning primitives before integrating with ROS publishers.
- Structure ROS integration tests under `src/trajectory_server/test/` using `ament_add_gtest`; gate them behind `colcon test`.
- For Python utilities, mirror pyproject settings: place files under `vtol_rl/tests/test_*.py` and name test classes `Test...`. Aim for smoke tests that validate casadi-generated models.
- Record new sample telemetry or YAML fixtures under `src/trajectory_server/config/samples/` to keep regression inputs versioned.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (`Add planner warm start`). Group related changes and rebase before pushing.
- Reference tracked issues in the body when closing work (`Refs #12`), and describe testing evidence (`Tested: colcon build && pytest`).
- PRs should include context, configuration updates touched, required launch instructions, and RViz or log screenshots when behaviour changes.

## Environment & Configuration Tips
- Use `install-ros.sh` on a fresh machine to install ROS Humble and tooling; verify locale before proceeding.
- Keep `global.toml` consistent with firmware expectations—document any field changes in the PR description.
- When adding new external solvers, update `pnc_setup.bash` exports and note additional dependencies in `pyproject.toml`.
