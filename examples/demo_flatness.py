"""Demonstrate on-demand CasADi flatness with custom quadrotor parameters.

Usage:
    uv run python examples/demo_flatness.py
    uv run minco-demo-flatness
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    import minco.flatness_cache

    params = {
        "mass": 1.1,
        "gravity": 9.81,
        "horizontal_drag": 0.05,
        "vertical_drag": 0.05,
        "parasitic_drag": 0.01,
        "speed_smooth": 0.001,
    }
    print("Creating CachedFlatness with parameters:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    cf = minco.flatness_cache.CachedFlatness(**params)

    vel = np.array([1.0, 0.5, 0.2])
    acc = np.array([0.1, 0.05, 0.02])
    jer = np.array([0.01, 0.005, 0.002])
    yaw = 0.3
    yaw_rate = 0.05

    thrust, quat, omg = cf.forward(vel, acc, jer, yaw, yaw_rate)
    print(f"\nForward (vel={vel}, acc={acc}, yaw={yaw:.2f}):")
    print(f"  thrust         = {thrust[0]:.4f} N")
    print(f"  quaternion     = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"  angular vel    = [{omg[0]:.6f}, {omg[1]:.6f}, {omg[2]:.6f}] rad/s")

    pos_grad = np.array([0.1, 0.2, 0.3])
    vel_grad = np.array([0.05, 0.1, 0.15])
    thr_grad = 0.5
    quat_grad = np.array([0.01, 0.02, 0.03, 0.04])
    omg_grad = np.array([0.1, 0.2, 0.3])

    pg, vg, ag, jg, psig, dpsig = cf.backward(pos_grad, vel_grad, thr_grad, quat_grad, omg_grad)
    print(f"\nBackward (adjoint-mode gradients):")
    print(f"  position total grad  = [{pg[0]:.6f}, {pg[1]:.6f}, {pg[2]:.6f}]")
    print(f"  velocity total grad  = [{vg[0]:.6f}, {vg[1]:.6f}, {vg[2]:.6f}]")
    print(f"  acceleration grad    = [{ag[0]:.6f}, {ag[1]:.6f}, {ag[2]:.6f}]")
    print(f"  jerk grad            = [{jg[0]:.6f}, {jg[1]:.6f}, {jg[2]:.6f}]")
    print(f"  yaw grad             = {psig:.6f}")
    print(f"  yaw rate grad        = {dpsig:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
