"""CasADi-based flatness generator for quadrotor dynamics.

This module constructs forward and backward flatness mappings that satisfy the
`minco::flatness::FlatnessModel` interface expectations while leveraging
CasADi for automatic differentiation. Parameters are sourced from YAML configs
that mirror the default C++ implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray


Array = NDArray[np.float64]


def _to_vector(vector: ArrayLike, size: int, label: str) -> np.ndarray:
    """Return a 1-D numpy array of the requested size with float64 dtype."""
    array = np.asarray(vector, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}")
    return array


@dataclass(frozen=True)
class QuadrotorFlatnessConfig:
    """Physical parameters shared with the native flatness implementation."""

    mass: float = 1.0
    gravity: float = 9.81
    horizontal_drag: float = 0.0
    vertical_drag: float = 0.0
    parasitic_drag: float = 0.0
    speed_smooth: float = 1.0e-3

    @classmethod
    def from_yaml(cls, path: Path) -> QuadrotorFlatnessConfig:
        """Load configuration from a YAML file (optionally namespaced under `flatness`)."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Flatness config file not found: {path}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"Flatness config must be a mapping, got {type(raw)!r}")

        node: Any = raw.get("flatness", raw)
        if not isinstance(node, dict):
            raise ValueError("Flatness node must be a mapping of parameter names to values")

        defaults = cls()
        return cls(
            mass=float(node.get("mass", defaults.mass)),
            gravity=float(node.get("gravity", defaults.gravity)),
            horizontal_drag=float(node.get("horizontal_drag", defaults.horizontal_drag)),
            vertical_drag=float(node.get("vertical_drag", defaults.vertical_drag)),
            parasitic_drag=float(node.get("parasitic_drag", defaults.parasitic_drag)),
            speed_smooth=float(node.get("speed_smooth", defaults.speed_smooth)),
        )

    def as_parameter_vector(self) -> np.ndarray:
        """Return parameters ordered to match the CasADi function signature."""
        return np.array(
            [
                self.mass,
                self.gravity,
                self.horizontal_drag,
                self.vertical_drag,
                self.parasitic_drag,
                self.speed_smooth,
            ],
            dtype=float,
        )


@dataclass
class ForwardQuery:
    velocity: Array
    acceleration: Array
    jerk: Array
    yaw: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class ForwardResult:
    thrust: float
    quaternion: Array
    angular_velocity: Array


@dataclass
class BackwardQuery:
    position_gradient: Array
    velocity_gradient: Array
    thrust_gradient: float
    quaternion_gradient: Array
    angular_velocity_gradient: Array
    velocity: Array | None = None
    acceleration: Array | None = None
    jerk: Array | None = None
    yaw: float | None = None
    yaw_rate: float | None = None


@dataclass
class BackwardResult:
    position_total_gradient: Array
    velocity_total_gradient: Array
    acceleration_total_gradient: Array
    jerk_total_gradient: Array
    yaw_total_gradient: float
    yaw_rate_total_gradient: float


class QuadrotorFlatnessGenerator:
    """CasADi-powered flatness map compatible with the GCOPTER flatness interface."""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_flatness_config.yaml"

    def __init__(self) -> None:
        self._config = QuadrotorFlatnessConfig()
        self._forward_func = self._build_forward_function()
        self._backward_func = self._build_backward_function(self._forward_func)
        self._last_forward_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None

    # ------------------------------------------------------------------
    # Configuration API -------------------------------------------------
    # ------------------------------------------------------------------
    def configure(self, config: QuadrotorFlatnessConfig) -> None:
        self._config = replace(config)

    def configure_from_file(self, file_path: str | None = None) -> None:
        path = Path(file_path) if file_path else self.DEFAULT_CONFIG_PATH
        self.configure(QuadrotorFlatnessConfig.from_yaml(path))

    def config(self) -> QuadrotorFlatnessConfig:
        return self._config

    # ------------------------------------------------------------------
    # Forward / backward evaluation ------------------------------------
    # ------------------------------------------------------------------
    def forward(self, query: ForwardQuery) -> ForwardResult:
        vel = _to_vector(query.velocity, 3, "velocity")
        acc = _to_vector(query.acceleration, 3, "acceleration")
        jer = _to_vector(query.jerk, 3, "jerk")
        params = self._config.as_parameter_vector()

        result = self._forward_func(
            vel,
            acc,
            jer,
            float(query.yaw),
            float(query.yaw_rate),
            *params,
        )

        # CasADi returns a list with a single vector output
        output = np.asarray(result[0], dtype=float).reshape(-1)
        thrust = float(output[0])
        quaternion = output[1:5]
        angular_velocity = output[5:8]

        self._last_forward_inputs = (
            vel.copy(),
            acc.copy(),
            jer.copy(),
            float(query.yaw),
            float(query.yaw_rate),
        )
        return ForwardResult(thrust=thrust, quaternion=quaternion, angular_velocity=angular_velocity)

    def backward(self, query: BackwardQuery) -> BackwardResult:
        vel, acc, jer, yaw, yaw_rate = self._resolve_backward_state(query)
        params = self._config.as_parameter_vector()

        pos_grad = _to_vector(query.position_gradient, 3, "position_gradient")
        vel_grad = _to_vector(query.velocity_gradient, 3, "velocity_gradient")
        quat_grad = _to_vector(query.quaternion_gradient, 4, "quaternion_gradient")
        omg_grad = _to_vector(query.angular_velocity_gradient, 3, "angular_velocity_gradient")

        result = self._backward_func(
            vel,
            acc,
            jer,
            yaw,
            yaw_rate,
            *params,
            pos_grad,
            vel_grad,
            float(query.thrust_gradient),
            quat_grad,
            omg_grad,
        )

        output = np.asarray(result[0], dtype=float).reshape(-1)
        position_total_grad = output[0:3]
        velocity_total_grad = output[3:6]
        acceleration_total_grad = output[6:9]
        jerk_total_grad = output[9:12]
        yaw_total_grad = float(output[12])
        yaw_rate_total_grad = float(output[13])

        return BackwardResult(
            position_total_gradient=position_total_grad,
            velocity_total_gradient=velocity_total_grad,
            acceleration_total_gradient=acceleration_total_grad,
            jerk_total_gradient=jerk_total_grad,
            yaw_total_gradient=yaw_total_grad,
            yaw_rate_total_gradient=yaw_rate_total_grad,
        )

    # ------------------------------------------------------------------
    # CasADi graph construction ----------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _build_forward_function() -> ca.Function:
        vel = ca.SX.sym("vel", 3)
        acc = ca.SX.sym("acc", 3)
        jer = ca.SX.sym("jer", 3)
        psi = ca.SX.sym("psi")
        dpsi = ca.SX.sym("dpsi")

        mass = ca.SX.sym("mass")
        grav = ca.SX.sym("grav")
        dh = ca.SX.sym("horizontal_drag")
        dv = ca.SX.sym("vertical_drag")
        cp = ca.SX.sym("parasitic_drag")
        veps = ca.SX.sym("speed_smooth")

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

        c_half_psi = ca.cos(0.5 * psi)
        s_half_psi = ca.sin(0.5 * psi)

        quat0 = tilt0 * c_half_psi
        quat1 = tilt1 * c_half_psi + tilt2 * s_half_psi
        quat2 = tilt2 * c_half_psi - tilt1 * s_half_psi
        quat3 = tilt0 * s_half_psi

        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)
        omg_den = z2 + 1.0
        omg_term = dz2 / omg_den

        omg0 = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term
        omg1 = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term
        omg2 = (z1 * dz0 - z0 * dz1) / omg_den + dpsi

        outputs = ca.vertcat(thrust, quat0, quat1, quat2, quat3, omg0, omg1, omg2)
        return ca.Function(
            "quadrotor_flatness_forward",
            [vel, acc, jer, psi, dpsi, mass, grav, dh, dv, cp, veps],
            [outputs],
            [
                "velocity",
                "acceleration",
                "jerk",
                "yaw",
                "yaw_rate",
                "mass",
                "gravity",
                "horizontal_drag",
                "vertical_drag",
                "parasitic_drag",
                "speed_smooth",
            ],
            ["flatness_outputs"],
        )

    @staticmethod
    def _build_backward_function(forward_func: ca.Function) -> ca.Function:
        vel = ca.SX.sym("vel", 3)
        acc = ca.SX.sym("acc", 3)
        jer = ca.SX.sym("jer", 3)
        psi = ca.SX.sym("psi")
        dpsi = ca.SX.sym("dpsi")

        mass = ca.SX.sym("mass")
        grav = ca.SX.sym("grav")
        dh = ca.SX.sym("horizontal_drag")
        dv = ca.SX.sym("vertical_drag")
        cp = ca.SX.sym("parasitic_drag")
        veps = ca.SX.sym("speed_smooth")

        pos_grad = ca.SX.sym("pos_grad", 3)
        vel_grad = ca.SX.sym("vel_grad", 3)
        thr_grad = ca.SX.sym("thr_grad")
        quat_grad = ca.SX.sym("quat_grad", 4)
        omg_grad = ca.SX.sym("omg_grad", 3)

        forward_out = forward_func(
            vel,
            acc,
            jer,
            psi,
            dpsi,
            mass,
            grav,
            dh,
            dv,
            cp,
            veps,
        )[0]

        jac_forward = ca.jacobian(
            forward_out,
            ca.vertcat(vel, acc, jer, psi, dpsi),
        )

        output_grad = ca.vertcat(thr_grad, quat_grad, omg_grad)
        input_grad = ca.vertcat(vel_grad, ca.SX.zeros(3), ca.SX.zeros(3), ca.SX.zeros(1), ca.SX.zeros(1))
        total_grad = ca.mtimes(jac_forward.T, output_grad) + input_grad

        vel_total_grad = total_grad[0:3]
        acc_total_grad = total_grad[3:6]
        jer_total_grad = total_grad[6:9]
        psi_total_grad = total_grad[9]
        dpsi_total_grad = total_grad[10]
        pos_total_grad = pos_grad

        outputs = ca.vertcat(
            pos_total_grad,
            vel_total_grad,
            acc_total_grad,
            jer_total_grad,
            psi_total_grad,
            dpsi_total_grad,
        )

        return ca.Function(
            "quadrotor_flatness_backward",
            [
                vel,
                acc,
                jer,
                psi,
                dpsi,
                mass,
                grav,
                dh,
                dv,
                cp,
                veps,
                pos_grad,
                vel_grad,
                thr_grad,
                quat_grad,
                omg_grad,
            ],
            [outputs],
            [
                "velocity",
                "acceleration",
                "jerk",
                "yaw",
                "yaw_rate",
                "mass",
                "gravity",
                "horizontal_drag",
                "vertical_drag",
                "parasitic_drag",
                "speed_smooth",
                "position_gradient",
                "velocity_gradient",
                "thrust_gradient",
                "quaternion_gradient",
                "angular_velocity_gradient",
            ],
            ["flatness_backward_outputs"],
        )

    # ------------------------------------------------------------------
    # Helpers -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _resolve_backward_state(self, query: BackwardQuery) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        if query.velocity is not None:
            vel = _to_vector(query.velocity, 3, "velocity")
        elif self._last_forward_inputs is not None:
            vel = self._last_forward_inputs[0]
        else:
            raise ValueError("Backward query missing velocity; call forward() first or provide velocity explicitly")

        if query.acceleration is not None:
            acc = _to_vector(query.acceleration, 3, "acceleration")
        elif self._last_forward_inputs is not None:
            acc = self._last_forward_inputs[1]
        else:
            raise ValueError("Backward query missing acceleration; call forward() first or provide acceleration explicitly")

        if query.jerk is not None:
            jer = _to_vector(query.jerk, 3, "jerk")
        elif self._last_forward_inputs is not None:
            jer = self._last_forward_inputs[2]
        else:
            raise ValueError("Backward query missing jerk; call forward() first or provide jerk explicitly")

        if query.yaw is not None:
            yaw = float(query.yaw)
        elif self._last_forward_inputs is not None:
            yaw = self._last_forward_inputs[3]
        else:
            raise ValueError("Backward query missing yaw; call forward() first or provide yaw explicitly")

        if query.yaw_rate is not None:
            yaw_rate = float(query.yaw_rate)
        elif self._last_forward_inputs is not None:
            yaw_rate = self._last_forward_inputs[4]
        else:
            raise ValueError("Backward query missing yaw_rate; call forward() first or provide yaw_rate explicitly")

        return vel, acc, jer, yaw, yaw_rate


__all__ = [
    "Array",
    "QuadrotorFlatnessConfig",
    "ForwardQuery",
    "ForwardResult",
    "BackwardQuery",
    "BackwardResult",
    "QuadrotorFlatnessGenerator",
]
