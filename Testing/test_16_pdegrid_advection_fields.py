"""Spatially varying PDE advection against independent conservative references."""
import numpy as np
import pytest


def _advect_periodic_1d(initial, face_vel, dt, dx):
    """First-order upwind finite-volume update from +face velocities."""
    f = np.asarray(initial, dtype=np.float64)
    v = np.asarray(face_vel, dtype=np.float64)
    out = f.copy()
    c = dt / dx
    n = len(f)
    # Face i is between cell i and i+1 (periodic).
    for i in range(n):
        j = (i + 1) % n
        flux = v[i] * (f[i] if v[i] >= 0 else f[j])
        out[i] -= c * flux
        out[j] += c * flux
    return out


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
def test_variable_advection_periodic_matches_independent_flux_reference(api, method):
    initial = np.array([1.0, 0.5, 2.0, 4.0, 1.5, 0.25], dtype=np.float32)
    values = np.array([0.3, -0.2, 0.4, 0.1, -0.35, 0.25], dtype=np.float32)
    dt, dx = 0.2, 0.8
    g = api.NewPDEgrid((-len(initial),))
    g[:] = initial
    g.SetTimeSpaceStep(dt, dx)
    getattr(g, method)(values)
    g.Update()
    if method == "AdvectionInterfaces":
        face = values.astype(np.float64)
    else:
        face = 0.5 * (values.astype(np.float64) + np.roll(values.astype(np.float64), -1))
    expected = _advect_periodic_1d(initial, face, dt, dx)
    np.testing.assert_allclose(g[:], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
def test_constant_spatial_velocity_equals_uniform_advection(api, method):
    rng = np.random.default_rng(9012)
    initial = rng.random(31, dtype=np.float32)
    velocity = -0.37
    a = api.NewPDEgrid((-31,))
    b = api.NewPDEgrid((-31,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(0.1, 0.7); b.SetTimeSpaceStep(0.1, 0.7)
    a.Advection(velocity)
    getattr(b, method)(np.full(31, velocity, dtype=np.float32))
    a.Update(); b.Update()
    np.testing.assert_allclose(a[:], b[:], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
def test_variable_periodic_advection_conserves_mass(api, method):
    rng = np.random.default_rng(9123)
    initial = rng.random(40, dtype=np.float32)
    velocity = rng.uniform(-0.25, 0.25, 40).astype(np.float32)
    g = api.NewPDEgrid((-40,))
    g[:] = initial
    g.SetTimeSpaceStep(0.1, 1.0)
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(50):
        getattr(g, method)(velocity)
        g.Update()
    assert float(np.sum(g[:], dtype=np.float64)) == pytest.approx(before, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
def test_safe_invalid_velocity_field_is_transactional(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode requires finite velocity fields")
    initial = np.arange(8, dtype=np.float32)
    g = api.NewPDEgrid((-8,))
    g[:] = initial
    velocity = np.full(8, 0.1, dtype=np.float32)
    velocity[4] = np.nan
    with pytest.raises(ValueError):
        getattr(g, method)(velocity)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
def test_safe_variable_advection_cfl_failure_is_transactional(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode assumes stable advection")
    initial = np.arange(8, dtype=np.float32)
    g = api.NewPDEgrid((-8,))
    g[:] = initial
    g.SetTimeSpaceStep(1.0, 1.0)
    velocity = np.full(8, 1.2, dtype=np.float32)
    with pytest.raises(ValueError):
        getattr(g, method)(velocity)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)
