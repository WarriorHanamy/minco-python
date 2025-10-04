from __future__ import annotations

from pathlib import Path

import numpy as np

import minco


def test_flatness_forward_backward_zero_motion(tmp_path: Path) -> None:
    config_file = tmp_path / "flatness.yaml"
    config_file.write_text(
        "\n".join(
            [
                "flatness:",
                "  mass: 1.0",
                "  gravity: 9.81",
                "  horizontal_drag: 0.1",
                "  vertical_drag: 0.1",
                "  parasitic_drag: 0.01",
                "  speed_smooth: 1e-3",
            ]
        ),
        encoding="utf-8",
    )

    mapper = minco.flatness.FlatnessMap()
    mapper.configure_from_file(str(config_file))

    vel = np.zeros(3)
    acc = np.zeros(3)
    jer = np.zeros(3)

    thrust, quat, omg = mapper.forward(vel, acc, jer, psi=0.0, dpsi=0.0)
    np.testing.assert_allclose(thrust, 9.81, atol=1e-6)
    np.testing.assert_allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(omg, np.zeros(3), atol=1e-6)

    outputs = mapper.backward(
        pos_grad=np.zeros(3),
        vel_grad=np.zeros(3),
        thr_grad=0.0,
        quat_grad=np.zeros(4),
        omg_grad=np.zeros(3),
    )

    for grad in outputs:
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)


def test_flatness_configure_from_file(tmp_path: Path) -> None:
    config_file = tmp_path / "flatness.yaml"
    config_file.write_text(
        "\n".join(
            [
                "# test flatness config",
                "mass: 2.0",
                "gravity: 9.81",
                "horizontal_drag: 0.05",
                "vertical_drag: 0.05",
                "parasitic_drag: 0.01",
                "speed_smooth: 1.0e-3",
            ]
        ),
        encoding="utf-8",
    )

    mapper = minco.flatness.FlatnessMap()
    mapper.configure_from_file(str(config_file))

    vel = np.zeros(3)
    acc = np.zeros(3)
    jer = np.zeros(3)

    thrust, _, _ = mapper.forward(vel, acc, jer, psi=0.0, dpsi=0.0)
    np.testing.assert_allclose(thrust, 2.0 * 9.81, atol=1e-6)


def test_flatness_default_config_file() -> None:
    mapper = minco.flatness.FlatnessMap()
    mapper.configure_from_file()

    thrust, _, _ = mapper.forward(
        np.zeros(3), np.zeros(3), np.zeros(3), psi=0.0, dpsi=0.0
    )

    np.testing.assert_allclose(thrust, 1.1 * 9.81, atol=1e-6)


if __name__ == "__main__":
    test_flatness_forward_backward_zero_motion()
