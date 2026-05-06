# API Reference

All public names are accessible via `import minco`.

---

## `minco.poly_traj`

Polynomial trajectory primitives.

### `Piece5` / `Piece7`

A single polynomial piece of degree `D` (5 or 7), spanning `[0, duration]`.
Spatial dimension is always 3.

| Signature | Returns | Notes |
|-----------|---------|-------|
| `PieceD()` | `PieceD` | Default constructor |
| `PieceD(duration, coeff_mat)` | `PieceD` | `duration: float`, `coeff_mat: np.ndarray` shape `(3, D+1)` |
| `.duration` | `float` | Read-only |
| `.degree` | `int` | Read-only, `D` |
| `.dim` | `int` | Read-only, always `3` |
| `.get_coeff_mat()` | `np.ndarray` | Copy of coefficient matrix `(3, D+1)` |
| `.get_pos(t)` | `np.ndarray` | Position `(3,)` at time `t` |
| `.get_vel(t)` | `np.ndarray` | Velocity `(3,)` at time `t` |
| `.get_acc(t)` | `np.ndarray` | Acceleration `(3,)` at time `t` |
| `.get_jer(t)` | `np.ndarray` | Jerk `(3,)` at time `t` |
| `.get_max_vel_rate()` | `float` | Max velocity magnitude in piece |
| `.get_max_acc_rate()` | `float` | Max acceleration magnitude in piece |
| `.check_max_vel_rate(v)` | `bool` | Whether max vel ≤ `v` |
| `.check_max_acc_rate(v)` | `bool` | Whether max acc ≤ `v` |

### `Trajectory5` / `Trajectory7`

A sequence of `PieceD` pieces forming a continuous trajectory. This is the
primary output of `GCOPTER.optimize()`.

```python
traj = GCOPTER.optimize(...)[1]  # returns Trajectory5
```

| Signature | Returns | Notes |
|-----------|---------|-------|
| `TrajectoryD()` | `TrajectoryD` | Empty trajectory |
| `TrajectoryD(durations, coeff_mats)` | `TrajectoryD` | `durations: list[float]`, `coeff_mats: list[np.ndarray]` |
| `.total_duration` | `float` | Sum of all piece durations |
| `.durations` | `np.ndarray` | `(N,)` durations of each piece |
| `.positions` | `np.ndarray` | `(3, N)` waypoint positions |
| `.get_piece_num()` | `int` | Number of pieces |
| `.get_pos(t)` | `np.ndarray` | Position `(3,)` at global time `t` |
| `.get_vel(t)` | `np.ndarray` | Velocity `(3,)` at global time `t` |
| `.get_acc(t)` | `np.ndarray` | Acceleration `(3,)` at global time `t` |
| `.get_jer(t)` | `np.ndarray` | Jerk `(3,)` at global time `t` |
| `.clear()` | `None` | Remove all pieces |
| `.append_piece(piece)` | `None` | Append a `PieceD` |
| `.get_max_vel_rate()` | `float` | Max velocity magnitude across all pieces |
| `.get_max_acc_rate()` | `float` | Max acceleration magnitude across all pieces |
| `len(traj)` | `int` | Number of pieces |
| `traj[i]` | `PieceD` | Access piece by index (supports `-1`) |
| `iter(traj)` | iterator | Iterate over pieces |

---

## `minco.sdlp`

Small-Dimensional Linear Programming solver (dimensions 1–6).

```python
minimum, argmin = minco.sdlp.linprog(c, A, b)
```

| Signature | Returns | Notes |
|-----------|---------|-------|
| `linprog(c, A, b)` | `tuple[float, np.ndarray]` | `min c·x` s.t. `Ax ≤ b`. `c: (n,)`, `A: (m, n)`, `b: (m,)` |

---

## `minco.root_finder`

Polynomial root isolation and evaluation.

| Signature | Returns | Notes |
|-----------|---------|-------|
| `.highest_order` | `int` | Max polynomial order supported |
| `poly_conv(lhs, rhs)` | `np.ndarray` | Convolve two coefficient vectors |
| `poly_sqr(coeffs)` | `np.ndarray` | Self-convolution |
| `poly_val(coeffs, x, stability=True)` | `float` | Evaluate polynomial at scalar `x` |
| `count_roots(coeffs, lbound, ubound)` | `int` | Count distinct real roots in `(lbound, ubound)` |
| `solve_polynomial(coeffs, lbound, ubound, tol, isolation=True)` | `list[float]` | Real roots in interval, sorted |

Coefficient vectors use highest-degree-first ordering.

---

## `minco.geo_utils`

Half-space polytope geometry helpers. All polytopes are `(M, 4)` arrays where
each row is `[A, B, C, D]` encoding `Ax + By + Cz + D ≤ 0`.

| Signature | Returns | Notes |
|-----------|---------|-------|
| `find_interior(h_poly)` | `tuple[bool, np.ndarray]` | `(success, interior_point(3,))` |
| `overlap(left, right, eps=1e-6)` | `bool` | Whether two polytopes intersect |
| `enumerate_vertices(h_poly, interior, eps=1e-6)` | `np.ndarray` | `(3, V)` vertex matrix |
| `enumerate_vertices_auto(h_poly, eps=1e-6)` | `tuple[bool, np.ndarray]` | Finds interior first, returns `(success, vertices)` |

---

## `minco.flatness`

Differential flatness maps for quadrotor dynamics.

### `FlatnessMap`

Hand-coded analytic flatness, runtime-configurable via YAML.

```python
fm = minco.flatness.FlatnessMap()
fm.configure_from_file("config/default_flatness_config.yaml")
thrust, quat, body_rates = fm.forward(vel, acc, jer, yaw, yaw_rate)
grads = fm.backward(pos_grad, vel_grad, thrust_grad, quat_grad, body_rate_grad)
```

| Signature | Returns | Notes |
|-----------|---------|-------|
| `FlatnessMap()` | `FlatnessMap` | Default constructor |
| `.configure_from_file(path="")` | `None` | Load YAML config; empty string uses embedded default |
| `.forward(vel, acc, jer, yaw, yaw_rate)` | `tuple[float, np.ndarray, np.ndarray]` | `vel, acc, jer: (3,)`. Returns `(thrust, quat[4], body_rates[3])` |
| `.backward(pos_grad, vel_grad, thrust_grad, quat_grad, body_rate_grad)` | `tuple[np.ndarray, ...]` | Adjoint map. Returns `(pos_total, vel_total, acc_total, jer_total, yaw_total, yaw_rate_total)` all `(3,)` except scalars |

### `CasadiQuadrotorFlatnessMap`

CasADi-generated flatness with embedded parameters (compile-time). Same
`forward`/`backward` signatures as `FlatnessMap`. Not runtime-configurable;
use `python -m minco.flatness_cache --generate-c` to re-embed new parameters.

### `CachedFlatness` (B2 — `minco.flatness_cache`)

On-demand CasADi flatness with disk caching.  Parameters are hashed; the first
call compiles via CasADi + gcc, subsequent calls load from
`~/.cache/minco/flatness/<sha256>.so`.

```python
from minco.flatness_cache import CachedFlatness

cf = CachedFlatness(mass=1.1, gravity=9.81, horizontal_drag=0.05,
                    vertical_drag=0.05, parasitic_drag=0.01, speed_smooth=0.001)
thrust, quat, omg = cf.forward(vel, acc, jer, yaw, yaw_rate)
pg, vg, ag, jg, psig, dpsig = cf.backward(pos_grad, vel_grad, thrust_grad,
                                           quat_grad, body_rate_grad)
```

| Signature | Returns | Notes |
|-----------|---------|-------|
| `CachedFlatness(mass, gravity, horizontal_drag, vertical_drag, parasitic_drag, speed_smooth)` | `CachedFlatness` | All params default to the embedded config values |
| `.forward(vel, acc, jer, yaw, yaw_rate)` | `tuple[np.ndarray, np.ndarray, np.ndarray]` | `(thrust[1], quat[4], body_rates[3])` |
| `.backward(pos_grad, vel_grad, thrust_grad, quat_grad, body_rate_grad)` | `tuple[np.ndarray, ...]` | Same semantics as `FlatnessMap.backward()` |

---

## `minco.gcopter`

Trajectory optimization with geometric control and safe-flight corridors.

### `GCOPTERPolytopeSFC`

Uses the analytic `FlatnessMap` (runtime-configurable).

```python
opt = minco.gcopter.GCOPTERPolytopeSFC()
opt.configure_from_file("config/default_gcopter.yaml")
```

| Signature | Returns | Notes |
|-----------|---------|-------|
| `GCOPTERPolytopeSFC()` | `GCOPTERPolytopeSFC` | |
| `.configure_from_file(path="")` | `None` | Load optimizer YAML config |
| `.setup_basic_trajectory(initial_pva, terminal_pva, initial_time, inner_points, sfc_polys, smoothing_factor, integral_resolution)` | `bool` | See below |
| `.optimize(rel_cost_tol)` | `tuple[float, Trajectory5]` | `(cost, trajectory)` |

### `GCOPTERPolytopeSFCCasadi`

Same API as `GCOPTERPolytopeSFC` but uses `CasadiQuadrotorFlatnessMap`
(compile-time embedded flatness parameters).

### `setup_basic_trajectory` Arguments

| Argument              | Shape          | Description                              |
|-----------------------|----------------|------------------------------------------|
| `initial_pva`         | `(3, 3)`       | Initial `[pos, vel, acc]` in columns       |
| `terminal_pva`        | `(3, 3)`       | Terminal `[pos, vel, acc]` in columns      |
| `initial_time`        | `(N,)`         | Per-piece initial durations [s]           |
| `inner_points`        | `(3, N-1)`     | Intermediate waypoint positions           |
| `sfc_polys`           | `list[np.ndarray]` | Safe-flight corridors, each `(M, 4)` half-space matrix |
| `smoothing_factor`    | `float`        | Smoothness regularization weight          |
| `integral_resolution` | `int`          | Integration resolution for cost           |

Returns `bool` — `True` on success.

---

## `minco.flatness_cache`

B2 module with CLI for cache management and developer code generation.

```bash
python -m minco.flatness_cache --list         # list cached parameter hashes
python -m minco.flatness_cache --clear        # clear all cached .so files
python -m minco.flatness_cache --info         # show cache directory + disk usage
python -m minco.flatness_cache --generate-c \ # regenerate embedded C sources
    --config config/casadi_quadrotor_flatness.yaml \
    --output-dir _src/minco_trajectory
```
