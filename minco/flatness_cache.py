"""B2 on-demand CasADi flatness compilation with disk caching.

Provides ``CachedFlatness``, a drop-in replacement for the compiled
``CasadiQuadrotorFlatnessMap`` that accepts arbitrary quadrotor parameters
at runtime.  The first instantiation with a given parameter set generates C
code via CasADi, compiles it with ``gcc -shared -fPIC -O2``, and stores the
result as ``~/.cache/minco/flatness/<sha256>.so``.  Subsequent instantiations
load the cached shared library directly (no CasADi or gcc needed).

CLI usage (``python -m minco.flatness_cache``):
  --list          List cached parameter sets
  --clear         Clear all cached .so files
  --info          Show cache directory and disk usage
  --generate-c    Regenerate embedded C sources (developer mode)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import ctypes
from ctypes import CDLL, POINTER, byref, c_double, c_int, c_longlong, cast
from pathlib import Path
from typing import Sequence

import casadi as ca
import numpy as np
import yaml
from numpy.typing import NDArray

PARAM_KEYS = (
    "mass",
    "gravity",
    "horizontal_drag",
    "vertical_drag",
    "parasitic_drag",
    "speed_smooth",
)

DEFAULT_PARAMS: dict[str, float] = {
    "mass": 1.1,
    "gravity": 9.81,
    "horizontal_drag": 0.05,
    "vertical_drag": 0.05,
    "parasitic_drag": 0.01,
    "speed_smooth": 0.001,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_SRC = PROJECT_ROOT / "_src" / "minco_trajectory" / "src" / "casadi_generated"
GENERATED_INCLUDE = PROJECT_ROOT / "_src" / "minco_trajectory" / "include" / "casadi_generated"


def _param_hash(params: dict[str, float]) -> str:
    """SHA256 hex digest of canonical JSON parameter dict."""
    canonical = json.dumps(
        {k: params[k] for k in PARAM_KEYS}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cache_dir() -> Path:
    """XDG-compliant cache directory."""
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base) / "minco" / "flatness"


def _build_functions(
    params: dict[str, float],
) -> tuple[ca.Function, ca.Function]:
    """Build CasADi forward and backward flatness functions for given params."""

    vel = ca.SX.sym("velocity", 3)
    acc = ca.SX.sym("acceleration", 3)
    jer = ca.SX.sym("jerk", 3)
    yaw = ca.SX.sym("yaw")
    yaw_rate = ca.SX.sym("yaw_rate")

    mass = ca.SX(params["mass"])
    grav = ca.SX(params["gravity"])
    dh = ca.SX(params["horizontal_drag"])
    dv = ca.SX(params["vertical_drag"])
    cp = ca.SX(params["parasitic_drag"])
    veps = ca.SX(params["speed_smooth"])

    v0, v1, v2 = vel[0], vel[1], vel[2]
    a0, a1, a2 = acc[0], acc[1], acc[2]
    j0, j1, j2 = jer[0], jer[1], jer[2]

    cp_term = ca.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + veps)
    w_term = 1.0 + cp * cp_term
    w0 = w_term * v0
    w1 = w_term * v1
    w2 = w_term * v2
    dh_over_m = dh / mass

    zu0 = a0 + dh_over_m * w0
    zu1 = a1 + dh_over_m * w1
    zu2 = a2 + dh_over_m * w2 + grav

    zu_sqr0 = zu0 * zu0
    zu_sqr1 = zu1 * zu1
    zu_sqr2 = zu2 * zu2
    zu01 = zu0 * zu1
    zu12 = zu1 * zu2
    zu02 = zu0 * zu2
    zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2
    zu_norm = ca.sqrt(zu_sqr_norm)

    z0 = zu0 / zu_norm
    z1 = zu1 / zu_norm
    z2 = zu2 / zu_norm

    ng_den = zu_sqr_norm * zu_norm
    ng00 = (zu_sqr1 + zu_sqr2) / ng_den
    ng01 = -zu01 / ng_den
    ng02 = -zu02 / ng_den
    ng11 = (zu_sqr0 + zu_sqr2) / ng_den
    ng12 = -zu12 / ng_den
    ng22 = (zu_sqr0 + zu_sqr1) / ng_den

    v_dot_a = v0 * a0 + v1 * a1 + v2 * a2
    dw_term = cp * v_dot_a / cp_term
    dw0 = w_term * a0 + dw_term * v0
    dw1 = w_term * a1 + dw_term * v1
    dw2 = w_term * a2 + dw_term * v2

    dz_term0 = j0 + dh_over_m * dw0
    dz_term1 = j1 + dh_over_m * dw1
    dz_term2 = j2 + dh_over_m * dw2

    dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2
    dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2
    dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2

    f_term0 = mass * a0 + dv * w0
    f_term1 = mass * a1 + dv * w1
    f_term2 = mass * (a2 + grav) + dv * w2

    thrust = z0 * f_term0 + z1 * f_term1 + z2 * f_term2

    tilt_den = ca.sqrt(2.0 * (1.0 + z2))
    tilt0 = 0.5 * tilt_den
    tilt1 = -z1 / tilt_den
    tilt2 = z0 / tilt_den

    c_half_psi = ca.cos(0.5 * yaw)
    s_half_psi = ca.sin(0.5 * yaw)

    quat0 = tilt0 * c_half_psi
    quat1 = tilt1 * c_half_psi + tilt2 * s_half_psi
    quat2 = tilt2 * c_half_psi - tilt1 * s_half_psi
    quat3 = tilt0 * s_half_psi

    c_psi = ca.cos(yaw)
    s_psi = ca.sin(yaw)
    omg_den = z2 + 1.0
    omg_term = dz2 / omg_den

    omg0 = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term
    omg1 = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term
    omg2 = (z1 * dz0 - z0 * dz1) / omg_den + yaw_rate

    forward_out = ca.vertcat(thrust, quat0, quat1, quat2, quat3, omg0, omg1, omg2)

    forward = ca.Function(
        "casadi_quadrotor_flatness_forward",
        [vel, acc, jer, yaw, yaw_rate],
        [forward_out],
        ["velocity", "acceleration", "jerk", "yaw", "yaw_rate"],
        ["flatness_outputs"],
    )

    pos_grad = ca.SX.sym("position_gradient", 3)
    vel_grad = ca.SX.sym("velocity_gradient", 3)
    thr_grad = ca.SX.sym("thrust_gradient")
    quat_grad = ca.SX.sym("quaternion_gradient", 4)
    omg_grad = ca.SX.sym("angular_velocity_gradient", 3)

    state = ca.vertcat(vel, acc, jer, yaw, yaw_rate)
    jac_forward = ca.jacobian(forward_out, state)

    output_grad = ca.vertcat(thr_grad, quat_grad, omg_grad)
    input_grad = ca.vertcat(
        vel_grad,
        ca.SX.zeros(3),
        ca.SX.zeros(3),
        ca.SX.zeros(1),
        ca.SX.zeros(1),
    )

    total_grad = ca.mtimes(jac_forward.T, output_grad) + input_grad

    backward_out = ca.vertcat(
        pos_grad,
        total_grad[0:3],
        total_grad[3:6],
        total_grad[6:9],
        total_grad[9],
        total_grad[10],
    )

    backward = ca.Function(
        "casadi_quadrotor_flatness_backward",
        [
            vel,
            acc,
            jer,
            yaw,
            yaw_rate,
            pos_grad,
            vel_grad,
            thr_grad,
            quat_grad,
            omg_grad,
        ],
        [backward_out],
        [
            "velocity",
            "acceleration",
            "jerk",
            "yaw",
            "yaw_rate",
            "position_gradient",
            "velocity_gradient",
            "thrust_gradient",
            "quaternion_gradient",
            "angular_velocity_gradient",
        ],
        ["flatness_backward_outputs"],
    )

    return forward, backward


def _write_config_header(params: dict[str, float], include_dir: Path) -> None:
    """Write quadrotor_flatness_config.hpp with embedded parameters."""
    include_dir.mkdir(parents=True, exist_ok=True)
    header_path = include_dir / "quadrotor_flatness_config.hpp"
    template = (
        "#pragma once\n\n"
        '#include "flatness.hpp"\n\n'
        "namespace minco::flatness::casadi_generated\n{{\n\n"
        "inline constexpr DefaultConfig kEmbeddedConfig{{\n"
        "    .mass            = {mass:.17g},\n"
        "    .gravity         = {gravity:.17g},\n"
        "    .horizontal_drag = {horizontal_drag:.17g},\n"
        "    .vertical_drag   = {vertical_drag:.17g},\n"
        "    .parasitic_drag  = {parasitic_drag:.17g},\n"
        "    .speed_smooth    = {speed_smooth:.17g},\n"
        "}};\n\n"
        "}}  // namespace minco::flatness::casadi_generated\n\n"
    )
    header_path.write_text(
        template.format(
            mass=params["mass"],
            gravity=params["gravity"],
            horizontal_drag=params["horizontal_drag"],
            vertical_drag=params["vertical_drag"],
            parasitic_drag=params["parasitic_drag"],
            speed_smooth=params["speed_smooth"],
        ),
        encoding="utf-8",
    )


def _generate_c_sources(
    params: dict[str, float],
    output_dir: Path,
) -> None:
    """Generate C source and header files via CasADi CodeGenerator."""
    forward, backward = _build_functions(params)

    src_dir = output_dir / "src" / "casadi_generated"
    include_dir = output_dir / "include" / "casadi_generated"
    src_dir.mkdir(parents=True, exist_ok=True)

    _write_config_header(params, include_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        codegen = ca.CodeGenerator("quadrotor_flatness.c", {"with_header": True})
        codegen.add(forward)
        codegen.add(backward)
        codegen.generate(str(tmp_path) + "/")

        shutil.copy2(tmp_path / "quadrotor_flatness.c", src_dir / "quadrotor_flatness.c")
        shutil.copy2(tmp_path / "quadrotor_flatness.h", include_dir / "quadrotor_flatness.h")


def _compile_so(c_source: Path, so_path: Path) -> None:
    """Compile C source to a shared library with gcc."""
    so_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            str(so_path),
            str(c_source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcc compilation failed:\n{result.stderr}\n{result.stdout}")


def _setup_ctypes_signatures(lib: CDLL) -> None:
    """Configure ctypes function signatures for the loaded CasADi shared library."""

    c_double_p = POINTER(c_double)
    c_longlong_p = POINTER(c_longlong)

    lib.casadi_quadrotor_flatness_forward.argtypes = [
        POINTER(c_double_p),
        POINTER(c_double_p),
        c_longlong_p,
        c_double_p,
        c_int,
    ]
    lib.casadi_quadrotor_flatness_forward.restype = c_int

    lib.casadi_quadrotor_flatness_forward_work.argtypes = [
        c_longlong_p,
        c_longlong_p,
        c_longlong_p,
        c_longlong_p,
    ]
    lib.casadi_quadrotor_flatness_forward_work.restype = c_int

    lib.casadi_quadrotor_flatness_forward_alloc_mem.argtypes = []
    lib.casadi_quadrotor_flatness_forward_alloc_mem.restype = c_int

    lib.casadi_quadrotor_flatness_forward_init_mem.argtypes = [c_int]
    lib.casadi_quadrotor_flatness_forward_init_mem.restype = c_int

    lib.casadi_quadrotor_flatness_forward_free_mem.argtypes = [c_int]
    lib.casadi_quadrotor_flatness_forward_free_mem.restype = None

    lib.casadi_quadrotor_flatness_backward.argtypes = [
        POINTER(c_double_p),
        POINTER(c_double_p),
        c_longlong_p,
        c_double_p,
        c_int,
    ]
    lib.casadi_quadrotor_flatness_backward.restype = c_int

    lib.casadi_quadrotor_flatness_backward_work.argtypes = [
        c_longlong_p,
        c_longlong_p,
        c_longlong_p,
        c_longlong_p,
    ]
    lib.casadi_quadrotor_flatness_backward_work.restype = c_int

    lib.casadi_quadrotor_flatness_backward_alloc_mem.argtypes = []
    lib.casadi_quadrotor_flatness_backward_alloc_mem.restype = c_int

    lib.casadi_quadrotor_flatness_backward_init_mem.argtypes = [c_int]
    lib.casadi_quadrotor_flatness_backward_init_mem.restype = c_int

    lib.casadi_quadrotor_flatness_backward_free_mem.argtypes = [c_int]
    lib.casadi_quadrotor_flatness_backward_free_mem.restype = None


class _FlatnessLib:
    """Low-level ctypes wrapper around a compiled CasADi flatness .so."""

    def __init__(self, so_path: Path) -> None:
        self._lib = CDLL(str(so_path))
        _setup_ctypes_signatures(self._lib)

        self._fwd_arg_sz = c_longlong()
        self._fwd_res_sz = c_longlong()
        self._fwd_iw_sz = c_longlong()
        self._fwd_w_sz = c_longlong()
        self._lib.casadi_quadrotor_flatness_forward_work(
            self._fwd_arg_sz, self._fwd_res_sz, self._fwd_iw_sz, self._fwd_w_sz
        )

        self._bwd_arg_sz = c_longlong()
        self._bwd_res_sz = c_longlong()
        self._bwd_iw_sz = c_longlong()
        self._bwd_w_sz = c_longlong()
        self._lib.casadi_quadrotor_flatness_backward_work(
            self._bwd_arg_sz, self._bwd_res_sz, self._bwd_iw_sz, self._bwd_w_sz
        )

        self._fwd_mem = self._lib.casadi_quadrotor_flatness_forward_alloc_mem()
        self._lib.casadi_quadrotor_flatness_forward_init_mem(self._fwd_mem)

        self._bwd_mem = self._lib.casadi_quadrotor_flatness_backward_alloc_mem()
        self._lib.casadi_quadrotor_flatness_backward_init_mem(self._bwd_mem)

    def forward(
        self, vel: np.ndarray, acc: np.ndarray, jer: np.ndarray, yaw: float, yaw_rate: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vel_ct = vel.ctypes.data_as(POINTER(c_double))
        acc_ct = acc.ctypes.data_as(POINTER(c_double))
        jer_ct = jer.ctypes.data_as(POINTER(c_double))
        yaw_arr = (c_double * 1)(yaw)
        dpsi_arr = (c_double * 1)(yaw_rate)

        arg_ptrs = (POINTER(c_double) * 5)()
        arg_ptrs[0] = vel_ct
        arg_ptrs[1] = acc_ct
        arg_ptrs[2] = jer_ct
        arg_ptrs[3] = cast(yaw_arr, POINTER(c_double))
        arg_ptrs[4] = cast(dpsi_arr, POINTER(c_double))

        buf = (c_double * 8)()
        res_ptrs = (POINTER(c_double) * 1)()
        res_ptrs[0] = cast(buf, POINTER(c_double))

        iw = (c_longlong * self._fwd_iw_sz.value)()
        w = (c_double * self._fwd_w_sz.value)()

        status = self._lib.casadi_quadrotor_flatness_forward(
            arg_ptrs, res_ptrs, iw, w, self._fwd_mem
        )
        if status != 0:
            raise RuntimeError(f"forward evaluation failed (status={status})")

        thrust = np.array([buf[0]], dtype=np.float64)
        quat = np.array([buf[1], buf[2], buf[3], buf[4]], dtype=np.float64)
        omg = np.array([buf[5], buf[6], buf[7]], dtype=np.float64)
        return thrust, quat, omg

    def backward(
        self,
        pos_grad: np.ndarray,
        vel_grad: np.ndarray,
        thr_grad: float,
        quat_grad: np.ndarray,
        omg_grad: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        jer: np.ndarray,
        yaw: float,
        yaw_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        vel_ct = vel.ctypes.data_as(POINTER(c_double))
        acc_ct = acc.ctypes.data_as(POINTER(c_double))
        jer_ct = jer.ctypes.data_as(POINTER(c_double))
        yaw_arr = (c_double * 1)(yaw)
        dpsi_arr = (c_double * 1)(yaw_rate)
        pos_ct = pos_grad.ctypes.data_as(POINTER(c_double))
        velg_ct = vel_grad.ctypes.data_as(POINTER(c_double))
        thr_arr = (c_double * 1)(thr_grad)
        quat_ct = quat_grad.ctypes.data_as(POINTER(c_double))
        omg_ct = omg_grad.ctypes.data_as(POINTER(c_double))

        arg_ptrs = (POINTER(c_double) * 10)()
        arg_ptrs[0] = vel_ct
        arg_ptrs[1] = acc_ct
        arg_ptrs[2] = jer_ct
        arg_ptrs[3] = cast(yaw_arr, POINTER(c_double))
        arg_ptrs[4] = cast(dpsi_arr, POINTER(c_double))
        arg_ptrs[5] = pos_ct
        arg_ptrs[6] = velg_ct
        arg_ptrs[7] = cast(thr_arr, POINTER(c_double))
        arg_ptrs[8] = quat_ct
        arg_ptrs[9] = omg_ct

        buf = (c_double * 14)()
        res_ptrs = (POINTER(c_double) * 1)()
        res_ptrs[0] = cast(buf, POINTER(c_double))

        iw = (c_longlong * self._bwd_iw_sz.value)()
        w = (c_double * self._bwd_w_sz.value)()

        status = self._lib.casadi_quadrotor_flatness_backward(
            arg_ptrs, res_ptrs, iw, w, self._bwd_mem
        )
        if status != 0:
            raise RuntimeError(f"backward evaluation failed (status={status})")

        pos_total = np.array([buf[0], buf[1], buf[2]], dtype=np.float64)
        vel_total = np.array([buf[3], buf[4], buf[5]], dtype=np.float64)
        acc_total = np.array([buf[6], buf[7], buf[8]], dtype=np.float64)
        jer_total = np.array([buf[9], buf[10], buf[11]], dtype=np.float64)
        psi_total = float(buf[12])
        dpsi_total = float(buf[13])
        return pos_total, vel_total, acc_total, jer_total, psi_total, dpsi_total

    def __del__(self) -> None:
        try:
            if hasattr(self, "_lib") and self._lib is not None:
                if hasattr(self, "_fwd_mem"):
                    self._lib.casadi_quadrotor_flatness_forward_free_mem(self._fwd_mem)
                if hasattr(self, "_bwd_mem"):
                    self._lib.casadi_quadrotor_flatness_backward_free_mem(self._bwd_mem)
        except Exception:
            pass


class CachedFlatness:
    """On-demand CasADi flatness with disk caching.

    Parameters are hashed; the compiled ``.so`` is cached in
    ``~/.cache/minco/flatness/<sha256>.so``.  The first call compiles
    with CasADi + gcc; subsequent calls load from cache instantly.
    """

    def __init__(
        self,
        mass: float = 1.1,
        gravity: float = 9.81,
        horizontal_drag: float = 0.05,
        vertical_drag: float = 0.05,
        parasitic_drag: float = 0.01,
        speed_smooth: float = 0.001,
    ) -> None:
        params: dict[str, float] = {
            "mass": mass,
            "gravity": gravity,
            "horizontal_drag": horizontal_drag,
            "vertical_drag": vertical_drag,
            "parasitic_drag": parasitic_drag,
            "speed_smooth": speed_smooth,
        }
        h = _param_hash(params)
        cache_dir = _cache_dir()
        so_path = cache_dir / f"{h}.so"

        if not so_path.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                c_path = Path(tmpdir) / "quadrotor_flatness.c"
                forward, backward = _build_functions(params)
                codegen = ca.CodeGenerator("quadrotor_flatness.c", {"with_header": True})
                codegen.add(forward)
                codegen.add(backward)
                codegen.generate(str(tmpdir) + "/")
                _compile_so(c_path, so_path)

        self._lib = _FlatnessLib(so_path)
        self._last_vel: np.ndarray | None = None
        self._last_acc: np.ndarray | None = None
        self._last_jer: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_yaw_rate: float | None = None

    def forward(
        self,
        vel: np.ndarray,
        acc: np.ndarray,
        jer: np.ndarray,
        yaw: float,
        yaw_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute flatness forward map.

        Args:
            vel: Velocity [3].
            acc: Acceleration [3].
            jer: Jerk [3].
            yaw: Yaw angle [rad].
            yaw_rate: Yaw rate [rad/s].

        Returns:
            (thrust, quaternion[4], angular_velocity[3]).
        """
        self._last_vel = np.asarray(vel, dtype=np.float64)
        self._last_acc = np.asarray(acc, dtype=np.float64)
        self._last_jer = np.asarray(jer, dtype=np.float64)
        self._last_yaw = float(yaw)
        self._last_yaw_rate = float(yaw_rate)
        return self._lib.forward(
            self._last_vel,
            self._last_acc,
            self._last_jer,
            self._last_yaw,
            self._last_yaw_rate,
        )

    def backward(
        self,
        pos_grad: np.ndarray,
        vel_grad: np.ndarray,
        thr_grad: float,
        quat_grad: np.ndarray,
        omg_grad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Compute flatness backward (adjoint) map.

        Requires a prior ``forward()`` call.
        """
        if self._last_vel is None:
            raise RuntimeError("CachedFlatness.backward() requires a prior forward() call")
        return self._lib.backward(
            np.asarray(pos_grad, dtype=np.float64),
            np.asarray(vel_grad, dtype=np.float64),
            float(thr_grad),
            np.asarray(quat_grad, dtype=np.float64),
            np.asarray(omg_grad, dtype=np.float64),
            self._last_vel,
            self._last_acc,
            self._last_jer,
            self._last_yaw,
            self._last_yaw_rate,
        )


def _cli_list() -> int:
    cache = _cache_dir()
    if not cache.exists():
        print(f"No cache directory at {cache}")
        return 0

    entries = sorted(cache.glob("*.so"))
    if not entries:
        print("No cached flatness libraries.")
        return 0

    print(f"{len(entries)} cached flatness libraries in {cache}:")
    for e in entries:
        size_mb = e.stat().st_size / (1024 * 1024)
        print(f"  {e.name}  ({size_mb:.1f} MB)")
    return 0


def _cli_clear() -> int:
    cache = _cache_dir()
    if cache.exists():
        count = len(list(cache.glob("*.so")))
        shutil.rmtree(cache)
        print(f"Cleared {count} cached flatness libraries from {cache}")
    else:
        print("No cache directory to clear.")
    return 0


def _cli_info() -> int:
    cache = _cache_dir()
    print(f"Cache directory: {cache}")
    if cache.exists():
        entries = list(cache.glob("*.so"))
        total_bytes = sum(e.stat().st_size for e in entries)
        print(f"Entries: {len(entries)}")
        print(f"Total size: {total_bytes / (1024 * 1024):.1f} MB")
    else:
        print("Cache directory does not exist yet.")
    return 0


def _cli_generate_c(config_path: Path | None, output_dir: Path | None) -> int:
    params = dict(DEFAULT_PARAMS)
    if config_path is not None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping, got {type(raw)!r}")
        node = raw.get("flatness", raw)
        if not isinstance(node, dict):
            raise ValueError("Flatness node must be a mapping")
        for key in PARAM_KEYS:
            if key in node:
                params[key] = float(node[key])

    od = output_dir or PROJECT_ROOT / "_src" / "minco_trajectory"
    _generate_c_sources(params, od)
    print(f"Generated CasADi C sources in {od}")
    print(f"  src/casadi_generated/quadrotor_flatness.c")
    print(f"  include/casadi_generated/quadrotor_flatness.h")
    print(f"  include/casadi_generated/quadrotor_flatness_config.hpp")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage cached CasADi flatness libraries.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List cached parameter sets")
    group.add_argument("--clear", action="store_true", help="Clear all cached .so files")
    group.add_argument("--info", action="store_true", help="Show cache directory info")
    group.add_argument(
        "--generate-c",
        action="store_true",
        help="Regenerate embedded C sources (developer mode)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config for --generate-c",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for --generate-c (default: _src/minco_trajectory)",
    )

    args = parser.parse_args(argv or sys.argv[1:])

    if args.list:
        return _cli_list()
    if args.clear:
        return _cli_clear()
    if args.info:
        return _cli_info()
    if args.generate_c:
        return _cli_generate_c(args.config, args.output_dir)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
