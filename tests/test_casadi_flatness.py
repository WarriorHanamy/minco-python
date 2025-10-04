"""
CasADi implementation of flatness forward function with automatic differentiation for backward.
This implementation replicates the functionality from flatness.hpp using CasADi symbolic computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import casadi as ca
import minco
import numpy as np
import pytest
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _write_flatness_config(tmp_path: Path, params: Dict[str, float]) -> Path:
    lines = ["flatness:"]
    for key in (
        "mass",
        "gravity",
        "horizontal_drag",
        "vertical_drag",
        "parasitic_drag",
        "speed_smooth",
    ):
        lines.append(f"  {key}: {params[key]}")

    path = tmp_path / "flatness.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _create_mappers(
    tmp_path: Path, params: Dict[str, float]
) -> Tuple[CasadiFlatnessMap, minco.flatness.FlatnessMap]:
    config_path = _write_flatness_config(tmp_path, params)
    casadi_mapper = CasadiFlatnessMap()
    casadi_mapper.reset(**params)

    reference_mapper = minco.flatness.FlatnessMap()
    reference_mapper.configure_from_file(str(config_path))
    return casadi_mapper, reference_mapper


class CasadiFlatnessMap:
    """CasADi implementation of the flatness mapping with automatic differentiation."""

    def __init__(self) -> None:
        self.mass = 1.0
        self.grav = 9.81
        self.dh = 0.0
        self.dv = 0.0
        self.cp = 0.0
        self.veps = 1.0e-3

        # Create symbolic variables for forward computation
        self.vel_sym = ca.SX.sym("vel", 3)
        self.acc_sym = ca.SX.sym("acc", 3)
        self.jer_sym = ca.SX.sym("jer", 3)
        self.psi_sym = ca.SX.sym("psi")
        self.dpsi_sym = ca.SX.sym("dpsi")

        # Create symbolic variables for parameters
        self.mass_sym = ca.SX.sym("mass")
        self.grav_sym = ca.SX.sym("grav")
        self.dh_sym = ca.SX.sym("dh")
        self.dv_sym = ca.SX.sym("dv")
        self.cp_sym = ca.SX.sym("cp")
        self.veps_sym = ca.SX.sym("veps")

        # Build forward function
        self.forward_func = self._build_forward_function()

        # Build backward function using automatic differentiation
        self.backward_func = self._build_backward_function()

    def reset(
        self,
        mass: float,
        gravity: float,
        horizontal_drag: float,
        vertical_drag: float,
        parasitic_drag: float,
        speed_smooth: float,
    ) -> None:
        """Reset the model parameters."""
        self.mass = mass
        self.grav = gravity
        self.dh = horizontal_drag
        self.dv = vertical_drag
        self.cp = parasitic_drag
        self.veps = speed_smooth

    def _build_forward_function(self) -> ca.Function:
        """Build the forward function using CasADi symbolic computation."""
        v0, v1, v2 = self.vel_sym[0], self.vel_sym[1], self.vel_sym[2]
        a0, a1, a2 = self.acc_sym[0], self.acc_sym[1], self.acc_sym[2]
        j0, j1, j2 = self.jer_sym[0], self.jer_sym[1], self.jer_sym[2]
        psi = self.psi_sym
        dpsi = self.dpsi_sym

        # Extract parameters
        mass = self.mass_sym
        grav = self.grav_sym
        dh = self.dh_sym
        dv = self.dv_sym
        cp = self.cp_sym
        veps = self.veps_sym

        # Implement the forward computation from flatness.hpp:229-303
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
        quat = ca.vertcat(quat0, quat1, quat2, quat3)

        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)
        omg_den = z2 + 1.0
        omg_term = dz2 / omg_den

        omg0 = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term
        omg1 = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term
        omg2 = (z1 * dz0 - z0 * dz1) / omg_den + dpsi
        omg = ca.vertcat(omg0, omg1, omg2)

        # Create the forward function
        inputs = ca.vertcat(
            self.vel_sym,
            self.acc_sym,
            self.jer_sym,
            self.psi_sym,
            self.dpsi_sym,
            self.mass_sym,
            self.grav_sym,
            self.dh_sym,
            self.dv_sym,
            self.cp_sym,
            self.veps_sym,
        )
        outputs = ca.vertcat(thrust, quat, omg)

        return ca.Function("forward", [inputs], [outputs], ["inputs"], ["outputs"])

    def _build_backward_function(self) -> ca.Function:
        """Build the backward function using CasADi automatic differentiation."""
        # Get the forward outputs
        forward_outputs = self.forward_func(
            ca.vertcat(
                self.vel_sym,
                self.acc_sym,
                self.jer_sym,
                self.psi_sym,
                self.dpsi_sym,
                self.mass_sym,
                self.grav_sym,
                self.dh_sym,
                self.dv_sym,
                self.cp_sym,
                self.veps_sym,
            )
        )

        # Extract thrust, quaternion, and angular velocity
        thrust = forward_outputs[0]
        quat = forward_outputs[1:5]
        omg = forward_outputs[5:8]

        # Create symbolic variables for gradients
        pos_grad_sym = ca.SX.sym("pos_grad", 3)
        vel_grad_sym = ca.SX.sym("vel_grad", 3)
        thr_grad_sym = ca.SX.sym("thr_grad")
        quat_grad_sym = ca.SX.sym("quat_grad", 4)
        omg_grad_sym = ca.SX.sym("omg_grad", 3)

        # Compute the total gradient using automatic differentiation
        # The gradient of the forward outputs w.r.t. inputs
        jac_forward = ca.jacobian(
            forward_outputs,
            ca.vertcat(
                self.vel_sym, self.acc_sym, self.jer_sym, self.psi_sym, self.dpsi_sym
            ),
        )

        # The backward gradients we want to compute
        output_grad = ca.vertcat(thr_grad_sym, quat_grad_sym, omg_grad_sym)

        # Compute the total gradients using the chain rule
        # total_grad = jac_forward^T * output_grad + input_grad
        input_grad = ca.vertcat(
            vel_grad_sym, ca.SX.zeros(3), ca.SX.zeros(3), ca.SX.zeros(1), ca.SX.zeros(1)
        )
        total_grad = ca.mtimes(jac_forward.T, output_grad) + input_grad

        # Extract the specific gradients we need
        vel_total_grad = total_grad[0:3]
        acc_total_grad = total_grad[3:6]
        jer_total_grad = total_grad[6:9]
        psi_total_grad = total_grad[9]
        dpsi_total_grad = total_grad[10]

        # Position gradient is passed through directly
        pos_total_grad = pos_grad_sym

        # Create the backward function
        inputs = ca.vertcat(
            self.vel_sym,
            self.acc_sym,
            self.jer_sym,
            self.psi_sym,
            self.dpsi_sym,
            self.mass_sym,
            self.grav_sym,
            self.dh_sym,
            self.dv_sym,
            self.cp_sym,
            self.veps_sym,
            pos_grad_sym,
            vel_grad_sym,
            thr_grad_sym,
            quat_grad_sym,
            omg_grad_sym,
        )
        outputs = ca.vertcat(
            pos_total_grad,
            vel_total_grad,
            acc_total_grad,
            jer_total_grad,
            psi_total_grad,
            dpsi_total_grad,
        )

        return ca.Function("backward", [inputs], [outputs], ["inputs"], ["outputs"])

    def forward(
        self,
        vel: Array,
        acc: Array,
        jer: Array,
        psi: float,
        dpsi: float,
    ) -> Tuple[float, Array, Array]:
        """Forward computation."""
        inputs = np.concatenate(
            [
                vel,
                acc,
                jer,
                [psi, dpsi],
                [self.mass, self.grav, self.dh, self.dv, self.cp, self.veps],
            ]
        )

        result = self.forward_func(inputs)
        thrust = float(result[0])
        quat = np.array(result[1:5], dtype=float).flatten()
        omg = np.array(result[5:8], dtype=float).flatten()

        return thrust, quat, omg

    def backward(
        self,
        pos_grad: Array,
        vel_grad: Array,
        thr_grad: float,
        quat_grad: Array,
        omg_grad: Array,
        vel: Array,
        acc: Array,
        jer: Array,
        psi: float,
        dpsi: float,
    ) -> Tuple[Array, Array, Array, Array, float, float]:
        """Backward computation using automatic differentiation."""
        inputs = np.concatenate(
            [
                vel,
                acc,
                jer,
                [psi, dpsi],
                [self.mass, self.grav, self.dh, self.dv, self.cp, self.veps],
                pos_grad,
                vel_grad,
                [thr_grad],
                quat_grad,
                omg_grad,
            ]
        )

        result = self.backward_func(inputs)

        pos_total_grad = np.array(result[0:3], dtype=float).flatten()
        vel_total_grad = np.array(result[3:6], dtype=float).flatten()
        acc_total_grad = np.array(result[6:9], dtype=float).flatten()
        jer_total_grad = np.array(result[9:12], dtype=float).flatten()
        psi_total_grad = float(result[12])
        dpsi_total_grad = float(result[13])

        return (
            pos_total_grad,
            vel_total_grad,
            acc_total_grad,
            jer_total_grad,
            psi_total_grad,
            dpsi_total_grad,
        )


def test_casadi_flatness_forward(tmp_path: Path | None = None) -> None:
    """Verify CasADi flatness forward outputs match the C++ bindings."""

    if tmp_path is None:
        tmp_path = Path("/tmp")
    test_params: Dict[str, float] = {
        "mass": 1.0,
        "gravity": 9.81,
        "horizontal_drag": 0.1,
        "vertical_drag": 0.1,
        "parasitic_drag": 0.01,
        "speed_smooth": 1.0e-3,
    }

    casadi_mapper, reference_mapper = _create_mappers(tmp_path, test_params)

    cases = [
        (
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            0.0,
            0.0,
        ),
        (
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.1, 0.0, 0.0], dtype=float),
            np.array([0.01, 0.0, 0.0], dtype=float),
            0.0,
            0.0,
        ),
        (
            np.array([0.5, 0.3, 0.1], dtype=float),
            np.array([0.2, 0.1, 0.05], dtype=float),
            np.array([0.01, 0.005, 0.002], dtype=float),
            0.5,
            0.1,
        ),
    ]

    for vel, acc, jer, psi, dpsi in cases:
        thrust_casadi, quat_casadi, omg_casadi = casadi_mapper.forward(
            vel, acc, jer, psi, dpsi
        )
        thrust_ref, quat_ref, omg_ref = reference_mapper.forward(
            vel, acc, jer, psi, dpsi
        )

        assert thrust_casadi == pytest.approx(thrust_ref, abs=1e-6)
        np.testing.assert_allclose(quat_casadi, quat_ref, atol=1e-6)
        np.testing.assert_allclose(omg_casadi, omg_ref, atol=1e-6)


def test_casadi_flatness_backward() -> None:
    """Verify CasADi backward gradients remain numerically stable."""
    test_params: Dict[str, float] = {
        "mass": 1.0,
        "gravity": 9.81,
        "horizontal_drag": 0.1,
        "vertical_drag": 0.1,
        "parasitic_drag": 0.01,
        "speed_smooth": 1.0e-3,
    }

    casadi_mapper = CasadiFlatnessMap()
    casadi_mapper.reset(**test_params)

    vel = np.array([0.5, 0.3, 0.1], dtype=float)
    acc = np.array([0.2, 0.1, 0.05], dtype=float)
    jer = np.array([0.01, 0.005, 0.002], dtype=float)
    psi = 0.5
    dpsi = 0.1

    pos_grad = np.array([0.1, 0.2, 0.3], dtype=float)
    vel_grad = np.array([0.05, 0.1, 0.15], dtype=float)
    thr_grad = 0.5
    quat_grad = np.array([0.01, 0.02, 0.03, 0.04], dtype=float)
    omg_grad = np.array([0.1, 0.2, 0.3], dtype=float)

    casadi_outputs = casadi_mapper.backward(
        pos_grad,
        vel_grad,
        thr_grad,
        quat_grad,
        omg_grad,
        vel,
        acc,
        jer,
        psi,
        dpsi,
    )

    expected_pos = np.array([0.1, 0.2, 0.3], dtype=float)
    expected_vel = np.array([0.05150152, 0.10063794, 0.20027595], dtype=float)
    expected_acc = np.array([0.01664616, 0.00607351, 0.49963408], dtype=float)
    expected_jer = np.array([0.02282376, 0.00043404, -0.00058452], dtype=float)
    expected_psi = 0.018642654194889696
    expected_dpsi = 0.3

    np.testing.assert_allclose(casadi_outputs[0], expected_pos, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(casadi_outputs[1], expected_vel, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(casadi_outputs[2], expected_acc, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(casadi_outputs[3], expected_jer, rtol=1e-6, atol=1e-8)
    assert casadi_outputs[4] == pytest.approx(expected_psi, rel=1e-6, abs=1e-8)
    assert casadi_outputs[5] == pytest.approx(expected_dpsi, rel=1e-6, abs=1e-8)


if __name__ == "__main__":
    print("Testing CasADi flatness implementation...")
    print("=" * 50)

    test_casadi_flatness_forward()
    print("\n" + "=" * 50)

    test_casadi_flatness_backward()
    print("\n" + "=" * 50)
