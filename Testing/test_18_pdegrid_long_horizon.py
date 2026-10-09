"""Long-horizon PDE scientific invariants and manufactured-solution checks."""
import math

import numpy as np
import pytest


@pytest.mark.parametrize("shape,spacing", [
    ((47,), (0.8,)),
    ((19, 23), (0.7, 1.1)),
    ((9, 11, 13), (0.8, 1.0, 1.3)),
])
def test_periodic_explicit_diffusion_preserves_mass_and_nonnegativity(api, shape, spacing):
    wrapped = tuple(-n for n in shape)
    rng = np.random.default_rng(18000 + len(shape))
    initial = rng.random(shape, dtype=np.float32)
    g = api.NewPDEgrid(wrapped)
    g[tuple(slice(None) for _ in shape)] = initial
    g.SetTimeSpaceStep(0.02, *(spacing + (1.0,) * (3 - len(shape))))
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(250):
        g.Diffusion(0.2)
        g.Update()
    final = g[tuple(slice(None) for _ in shape)]
    assert float(np.sum(final, dtype=np.float64)) == pytest.approx(before, rel=4e-5, abs=4e-5)
    assert np.min(final) >= -2e-6
    assert np.max(final) <= np.max(initial) + 2e-6


@pytest.mark.parametrize("n", [32, 64, 128])
def test_periodic_diffusion_fourier_mode_matches_discrete_exact_amplitude(api, n):
    # For the explicit scheme, a Fourier mode is an exact eigenvector. This
    # checks repeated stepping without comparing PAL to another PAL operator.
    k = 3
    rate, dt, dx, steps = 0.17, 0.03, 0.9, 80
    x = np.arange(n, dtype=np.float64)
    initial = np.sin(2 * math.pi * k * x / n).astype(np.float32)
    g = api.NewPDEgrid((-n,))
    g[:] = initial
    g.SetTimeSpaceStep(dt, dx)
    for _ in range(steps):
        g.Diffusion(rate)
        g.Update()
    lam = 1.0 - 4.0 * rate * dt / (dx * dx) * math.sin(math.pi * k / n) ** 2
    expected = initial.astype(np.float64) * lam**steps
    np.testing.assert_allclose(g[:], expected, rtol=3e-5, atol=3e-5)


def test_periodic_advection_fourier_mode_matches_discrete_exact_solution(api):
    # Complex amplification factor for first-order upwind with positive v:
    # G = 1-C + C exp(-i theta). Compare the real sinusoid after many steps.
    n, k, steps = 96, 5, 120
    velocity, dt, dx = 0.6, 0.04, 0.8
    courant = velocity * dt / dx
    theta = 2 * math.pi * k / n
    x = np.arange(n, dtype=np.float64)
    initial = np.sin(theta * x).astype(np.float32)
    g = api.NewPDEgrid((-n,))
    g[:] = initial
    g.SetTimeSpaceStep(dt, dx)
    for _ in range(steps):
        g.Advection(velocity)
        g.Update()
    amp = (1.0 - courant + courant * np.exp(-1j * theta)) ** steps
    expected = np.imag(np.exp(1j * theta * x) * amp)
    np.testing.assert_allclose(g[:], expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("method", ["DiffusionADI"])
def test_implicit_diffusion_large_timestep_remains_finite_and_smooths(api, method):
    rng = np.random.default_rng(18888)
    initial = rng.random((25, 27), dtype=np.float32)
    g = api.NewPDEgrid((-25, -27))
    g[:, :] = initial
    # Far beyond the explicit 2D stability limit, intentionally.
    g.SetTimeSpaceStep(2.0, 1.0, 1.0)
    before_var = float(np.var(initial, dtype=np.float64))
    before_mass = float(np.sum(initial, dtype=np.float64))
    for _ in range(20):
        getattr(g, method)(0.4)
        g.Update()
    final = g[:, :]
    assert np.all(np.isfinite(final))
    assert float(np.var(final, dtype=np.float64)) < before_var
    assert float(np.sum(final, dtype=np.float64)) == pytest.approx(before_mass, rel=3e-4, abs=3e-4)
