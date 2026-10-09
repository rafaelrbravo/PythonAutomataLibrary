"""PDEGrid numerical correctness against independent finite-difference references."""
import itertools

import numpy as np
import pytest


def _explicit_zero_flux(field, rate, dt, spacing):
    """One explicit diffusion step with zero-flux missing faces."""
    f = np.asarray(field, dtype=np.float64)
    out = f.copy()
    for idx in np.ndindex(f.shape):
        delta = 0.0
        for axis, dx in enumerate(spacing):
            s = rate * dt / (dx * dx)
            for sign in (-1, 1):
                nb = list(idx)
                nb[axis] += sign
                if 0 <= nb[axis] < f.shape[axis]:
                    delta += s * (f[tuple(nb)] - f[idx])
        out[idx] += delta
    return out


@pytest.mark.parametrize("shape", [(9,), (5, 6), (4, 3, 5)])
def test_geometry_and_storage(api, shape):
    g = api.NewPDEgrid(shape)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7
    g[tuple(slice(None) for _ in shape)] = values
    np.testing.assert_allclose(g[tuple(slice(None) for _ in shape)], values, rtol=0, atol=1e-7)
    for coordinates in itertools.product(*(range(n) for n in shape)):
        i = int(np.ravel_multi_index(coordinates, shape, order="C"))
        assert g.ToI(*coordinates) == i
        assert g.ItoX(i) == coordinates[0]
        if len(shape) > 1:
            assert g.ItoY(i) == coordinates[1]
        if len(shape) > 2:
            assert g.ItoZ(i) == coordinates[2]


def test_add_is_buffered_until_update(api):
    g = api.NewPDEgrid((5,))
    g[2] = 3.0
    g.Add(1.25, 2)
    assert g[2] == pytest.approx(3.0)
    g.Update()
    assert g[2] == pytest.approx(4.25)


@pytest.mark.parametrize("shape,spacing", [
    ((9,), (0.7,)),
    ((5, 6), (0.8, 1.3)),
    ((4, 3, 5), (0.9, 1.1, 1.4)),
])
def test_explicit_diffusion_one_step_matches_reference(api, shape, spacing):
    rng = np.random.default_rng(8128 + len(shape))
    initial = rng.normal(size=shape).astype(np.float32)
    dt, rate = 0.05, 0.4
    g = api.NewPDEgrid(shape)
    g[tuple(slice(None) for _ in shape)] = initial
    args = list(spacing) + [1.0] * (3 - len(spacing))
    g.SetTimeSpaceStep(dt, *args)
    g.Diffusion(rate)
    # Diffusion writes deltas; state changes only on Update.
    np.testing.assert_array_equal(g[tuple(slice(None) for _ in shape)], initial)
    g.Update()
    expected = _explicit_zero_flux(initial, rate, dt, spacing)
    np.testing.assert_allclose(g[tuple(slice(None) for _ in shape)], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("shape", [(31,), (13, 17), (7, 8, 9)])
def test_zero_flux_diffusion_conserves_mass(api, shape):
    rng = np.random.default_rng(1000 + len(shape))
    initial = rng.random(shape, dtype=np.float32)
    g = api.NewPDEgrid(shape)
    g[tuple(slice(None) for _ in shape)] = initial
    g.SetTimeSpaceStep(0.05, 1.0, 1.0, 1.0)
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(20):
        g.Diffusion(0.3)
        g.Update()
    after = float(np.sum(g[tuple(slice(None) for _ in shape)], dtype=np.float64))
    assert after == pytest.approx(before, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("shape", [(-31,), (-13, -17), (-7, -8, -9)])
def test_wrapped_diffusion_conserves_mass(api, shape):
    actual_shape = tuple(abs(x) for x in shape)
    rng = np.random.default_rng(2000 + len(shape))
    initial = rng.random(actual_shape, dtype=np.float32)
    g = api.NewPDEgrid(shape)
    g[tuple(slice(None) for _ in shape)] = initial
    g.SetTimeSpaceStep(0.04, 1.0, 1.0, 1.0)
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(25):
        g.Diffusion(0.25)
        g.Update()
    after = float(np.sum(g[tuple(slice(None) for _ in shape)], dtype=np.float64))
    assert after == pytest.approx(before, rel=2e-6, abs=2e-6)


def test_constant_field_is_fixed_point_zero_flux(api):
    g = api.NewPDEgrid((17, 19))
    g[:, :] = 4.125
    g.SetTimeSpaceStep(0.1, 0.7, 1.2)
    for _ in range(10):
        g.Diffusion(0.2)
        g.Update()
    np.testing.assert_array_equal(g[:, :], np.full((17, 19), 4.125, dtype=np.float32))


def test_uniform_advection_periodic_integer_cell_shift(api):
    # CFL=1 on a wrapped 1D grid: first-order upwind is exactly one-cell shift.
    initial = np.arange(11, dtype=np.float32)
    g = api.NewPDEgrid((-11,))
    g[:] = initial
    g.SetTimeSpaceStep(1.0, 1.0)
    g.Advection(1.0)
    g.Update()
    np.testing.assert_array_equal(g[:], np.roll(initial, 1))


def test_reset_clears_field_and_pending_delta(api):
    g = api.NewPDEgrid((8,))
    g[:] = np.arange(8, dtype=np.float32)
    g.Add(4.0, 3)
    g.Reset()
    np.testing.assert_array_equal(g[:], np.zeros(8, dtype=np.float32))
    g.Update()
    np.testing.assert_array_equal(g[:], np.zeros(8, dtype=np.float32))


def test_safe_rejects_unstable_explicit_diffusion_transactionally(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires caller to satisfy explicit stability condition")
    g = api.NewPDEgrid((9,))
    initial = np.arange(9, dtype=np.float32)
    g[:] = initial
    g.SetTimeSpaceStep(1.0, 1.0)
    with pytest.raises(ValueError):
        g.Diffusion(0.51)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)


def test_safe_rejects_advection_cfl_violation_transactionally(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires caller to satisfy CFL condition")
    g = api.NewPDEgrid((9,))
    initial = np.arange(9, dtype=np.float32)
    g[:] = initial
    g.SetTimeSpaceStep(1.0, 1.0)
    with pytest.raises(ValueError):
        g.Advection(1.01)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)


@pytest.mark.parametrize("args", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (np.nan, 1.0)])
def test_safe_rejects_invalid_time_space_steps(api, safe_mode, args):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise validation")
    g = api.NewPDEgrid((5,))
    with pytest.raises(ValueError):
        g.SetTimeSpaceStep(*args)
