"""Extended PDEGrid diffusion operators and boundary-condition tests.

References here are deliberately independent NumPy calculations for one-step
fluxes. Constant-rate equivalence cross-checks specialized PAL operators against
the already independently referenced scalar Diffusion path.
"""
import numpy as np
import pytest


def _harmonic(a, b):
    return 0.0 if a <= 0 or b <= 0 else 2.0 * a * b / (a + b)


def _field_diffusion_1d(initial, rates, dt, dx, wrapped=False):
    f = np.asarray(initial, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    out = f.copy()
    s = dt / (dx * dx)
    n = len(f)
    for i in range(n):
        for step in (-1, 1):
            j = i + step
            if wrapped:
                j %= n
            elif not 0 <= j < n:
                continue
            out[i] += _harmonic(r[i], r[j]) * s * (f[j] - f[i])
    return out


def test_scalar_dirichlet_boundary_flux_1d(api):
    initial = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    rate, dt, dx = 0.2, 0.1, 0.5
    g = api.NewPDEgrid((4,))
    g[:] = initial
    g.SetTimeSpaceStep(dt, dx)
    g.Diffusion(rate, xMinBC=10.0, xMaxBC=-2.0)
    g.Update()

    s = rate * dt / dx**2
    expected = initial.astype(np.float64)
    expected[0] += s * (initial[1] - initial[0]) + 2*s*(10.0 - initial[0])
    expected[1] += s * (initial[0] - initial[1]) + s*(initial[2] - initial[1])
    expected[2] += s * (initial[1] - initial[2]) + s*(initial[3] - initial[2])
    expected[3] += s * (initial[2] - initial[3]) + 2*s*(-2.0 - initial[3])
    np.testing.assert_allclose(g[:], expected, rtol=2e-6, atol=2e-6)


def test_face_array_boundary_condition_2d(api):
    initial = np.zeros((3, 4), dtype=np.float32)
    x_min = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    g = api.NewPDEgrid((3, 4))
    g[:, :] = initial
    g.SetTimeSpaceStep(0.1, 1.0, 1.0)
    g.Diffusion(0.25, xMinBC=x_min)
    g.Update()
    expected = np.zeros_like(initial)
    expected[0, :] = 2 * 0.1 * 0.25 * x_min
    np.testing.assert_allclose(g[:, :], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("wrapped", [False, True])
def test_diffusion_field_matches_harmonic_reference_1d(api, wrapped):
    dims = (-7,) if wrapped else (7,)
    initial = np.array([0.3, 1.1, -0.2, 2.0, 1.4, 0.7, 1.8], dtype=np.float32)
    rates = np.array([0.1, 0.3, 0.0, 0.25, 0.4, 0.15, 0.2], dtype=np.float32)
    dt, dx = 0.08, 0.7
    g = api.NewPDEgrid(dims)
    g[:] = initial
    g.SetTimeSpaceStep(dt, dx)
    g.DiffusionField(rates)
    g.Update()
    expected = _field_diffusion_1d(initial, rates, dt, dx, wrapped)
    np.testing.assert_allclose(g[:], expected, rtol=2e-6, atol=2e-6)


def test_constant_diffusion_field_equals_scalar_diffusion(api):
    rng = np.random.default_rng(3456)
    initial = rng.random((8, 9), dtype=np.float32)
    rate = 0.21
    scalar = api.NewPDEgrid((-8, -9))
    field = api.NewPDEgrid((-8, -9))
    scalar[:, :] = initial
    field[:, :] = initial
    scalar.SetTimeSpaceStep(0.07, 0.8, 1.1)
    field.SetTimeSpaceStep(0.07, 0.8, 1.1)
    scalar.Diffusion(rate)
    field.DiffusionField(np.full(initial.shape, rate, dtype=np.float32))
    scalar.Update(); field.Update()
    np.testing.assert_allclose(field[:, :], scalar[:, :], rtol=2e-6, atol=2e-6)


def test_constant_interface_rates_equal_scalar_diffusion(api):
    rng = np.random.default_rng(4567)
    initial = rng.random((6, 7), dtype=np.float32)
    rate = 0.18
    scalar = api.NewPDEgrid((-6, -7))
    interfaces = api.NewPDEgrid((-6, -7))
    scalar[:, :] = initial
    interfaces[:, :] = initial
    scalar.SetTimeSpaceStep(0.05, 0.9, 1.2)
    interfaces.SetTimeSpaceStep(0.05, 0.9, 1.2)
    scalar.Diffusion(rate)
    rates = np.full(initial.shape, rate, dtype=np.float32)
    interfaces.DiffusionInterfaces(rates, rates)
    scalar.Update(); interfaces.Update()
    np.testing.assert_allclose(interfaces[:, :], scalar[:, :], rtol=2e-6, atol=2e-6)


def test_diffusion_mask_blocks_flux_across_masked_sites(api):
    initial = np.array([0., 0., 10., 0., 0.], dtype=np.float32)
    mask = np.array([1., 1., -1., 1., 1.], dtype=np.float32)
    g = api.NewPDEgrid((5,))
    g[:] = initial
    g.SetTimeSpaceStep(0.1, 1.0)
    g.DiffusionMask(0.4, mask)
    g.Update()
    # Masked center is neither updated nor coupled to its neighbors.
    np.testing.assert_array_equal(g[:], initial)


def test_diffusion_mask_unmasked_matches_scalar(api):
    rng = np.random.default_rng(5678)
    initial = rng.random((6, 5), dtype=np.float32)
    a = api.NewPDEgrid((6, 5)); b = api.NewPDEgrid((6, 5))
    a[:, :] = initial; b[:, :] = initial
    a.SetTimeSpaceStep(0.04, 1.0, 1.0); b.SetTimeSpaceStep(0.04, 1.0, 1.0)
    a.Diffusion(0.2)
    b.DiffusionMask(0.2, np.ones(initial.shape, dtype=np.float32))
    a.Update(); b.Update()
    np.testing.assert_allclose(a[:, :], b[:, :], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("method", ["DiffusionField", "DiffusionInterfaces"])
def test_safe_invalid_rate_field_is_transactional(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode requires valid rate fields")
    initial = np.arange(7, dtype=np.float32)
    g = api.NewPDEgrid((7,))
    g[:] = initial
    g.SetTimeSpaceStep(0.1, 1.0)
    rates = np.full(7, 0.2, dtype=np.float32)
    rates[3] = np.nan
    with pytest.raises(ValueError):
        getattr(g, method)(rates)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)


def test_safe_negative_diffusion_field_is_transactional(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires valid rate fields")
    initial = np.arange(7, dtype=np.float32)
    g = api.NewPDEgrid((7,)); g[:] = initial; g.SetTimeSpaceStep(0.1, 1.0)
    rates = np.full(7, 0.2, dtype=np.float32); rates[2] = -0.1
    with pytest.raises(ValueError):
        g.DiffusionField(rates)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)
