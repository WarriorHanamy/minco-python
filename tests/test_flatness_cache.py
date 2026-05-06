"""Integration tests for B2 on-demand CasADi flatness caching."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import minco
import minco.flatness_cache

Array = NDArray[np.float64]


@pytest.fixture(scope="module")
def default_params() -> dict[str, float]:
    return {
        "mass": 1.1,
        "gravity": 9.81,
        "horizontal_drag": 0.05,
        "vertical_drag": 0.05,
        "parasitic_drag": 0.01,
        "speed_smooth": 0.001,
    }


def test_cached_flatness_forward_parity(default_params: dict[str, float]) -> None:
    """CachedFlatness forward must match compiled CasadiQuadrotorFlatnessMap."""
    cf = minco.flatness_cache.CachedFlatness(**default_params)
    ref = minco.flatness.CasadiQuadrotorFlatnessMap()

    vel = np.array([1.0, 0.3, 0.1], dtype=np.float64)
    acc = np.array([0.1, 0.05, 0.02], dtype=np.float64)
    jer = np.array([0.01, 0.005, 0.002], dtype=np.float64)
    yaw = 0.5
    dpsi = 0.1

    t_c, q_c, o_c = cf.forward(vel, acc, jer, yaw, dpsi)
    t_r, q_r, o_r = ref.forward(vel, acc, jer, yaw, dpsi)

    np.testing.assert_allclose(q_c, q_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(o_c, o_r, rtol=1e-12, atol=1e-12)


def test_cached_flatness_backward_parity(default_params: dict[str, float]) -> None:
    """CachedFlatness backward must match compiled CasadiQuadrotorFlatnessMap."""
    cf = minco.flatness_cache.CachedFlatness(**default_params)
    ref = minco.flatness.CasadiQuadrotorFlatnessMap()

    vel = np.array([1.0, 0.3, 0.1], dtype=np.float64)
    acc = np.array([0.1, 0.05, 0.02], dtype=np.float64)
    jer = np.array([0.01, 0.005, 0.002], dtype=np.float64)
    yaw = 0.5
    dpsi = 0.1

    cf.forward(vel, acc, jer, yaw, dpsi)
    ref.forward(vel, acc, jer, yaw, dpsi)

    pos_g = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    vel_g = np.array([0.05, 0.1, 0.15], dtype=np.float64)
    thr_g = 0.5
    quat_g = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    omg_g = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    pg_c, vg_c, ag_c, jg_c, psig_c, dpsig_c = cf.backward(pos_g, vel_g, thr_g, quat_g, omg_g)
    pg_r, vg_r, ag_r, jg_r, psig_r, dpsig_r = ref.backward(pos_g, vel_g, thr_g, quat_g, omg_g)

    np.testing.assert_allclose(pg_c, pg_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vg_c, vg_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ag_c, ag_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jg_c, jg_r, rtol=1e-12, atol=1e-12)
    assert psig_c == pytest.approx(psig_r, rel=1e-12, abs=1e-12)
    assert dpsig_c == pytest.approx(dpsig_r, rel=1e-12, abs=1e-12)


def test_cache_hit_avoids_recompilation(
    default_params: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second CachedFlatness instantiation loads from cache."""
    cf1 = minco.flatness_cache.CachedFlatness(**default_params)
    compiled_called = False
    import subprocess as _sp

    orig_run = _sp.run

    def fake_run(*args, **kwargs):
        nonlocal compiled_called
        compiled_called = True
        return orig_run(*args, **kwargs)

    monkeypatch.setattr(_sp, "run", fake_run)

    cf2 = minco.flatness_cache.CachedFlatness(**default_params)
    assert not compiled_called, "second instantiation should not trigger gcc"

    vel = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    acc = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    jer = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    t1, q1, o1 = cf1.forward(vel, acc, jer, 0.0, 0.0)
    t2, q2, o2 = cf2.forward(vel, acc, jer, 0.0, 0.0)
    np.testing.assert_array_equal(q1, q2)


def test_different_params_yield_different_hash() -> None:
    """Different parameters should produce different cache hashes."""
    from minco.flatness_cache import _param_hash

    h1 = _param_hash(
        {
            "mass": 1.0,
            "gravity": 9.81,
            "horizontal_drag": 0.0,
            "vertical_drag": 0.0,
            "parasitic_drag": 0.0,
            "speed_smooth": 1e-3,
        }
    )
    h2 = _param_hash(
        {
            "mass": 1.1,
            "gravity": 9.81,
            "horizontal_drag": 0.0,
            "vertical_drag": 0.0,
            "parasitic_drag": 0.0,
            "speed_smooth": 1e-3,
        }
    )
    assert h1 != h2


def test_forward_without_prior_forward_raises() -> None:
    """backward() without prior forward() should raise."""
    cf = minco.flatness_cache.CachedFlatness(
        mass=1.0,
        gravity=9.81,
        horizontal_drag=0.0,
        vertical_drag=0.0,
        parasitic_drag=0.0,
        speed_smooth=1e-3,
    )
    with pytest.raises(RuntimeError, match="prior forward"):
        cf.backward(
            np.zeros(3),
            np.zeros(3),
            0.0,
            np.zeros(4),
            np.zeros(3),
        )
