"""PDEGrid ADI/radial scientific invariants and convergence tests."""
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(-31,), (-13, -17), (-7, -8, -9)])
def test_adi_periodic_conserves_mass(api, shape):
    actual = tuple(abs(n) for n in shape)
    rng = np.random.default_rng(6000 + len(shape))
    initial = rng.random(actual, dtype=np.float32)
    g = api.NewPDEgrid(shape)
    key = tuple(slice(None) for _ in shape)
    g[key] = initial
    g.SetTimeSpaceStep(0.3, 0.8, 1.1, 1.3)
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(12):
        g.DiffusionADI(0.7)
        g.Update()
    after = float(np.sum(g[key], dtype=np.float64))
    assert after == pytest.approx(before, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("shape", [(19,), (9, 11), (6, 7, 8)])
def test_adi_zero_flux_constant_field_is_fixed_point(api, shape):
    g = api.NewPDEgrid(shape)
    key = tuple(slice(None) for _ in shape)
    g[key] = 2.75
    g.SetTimeSpaceStep(0.8, 0.7, 1.2, 1.4)
    for _ in range(8):
        g.DiffusionADI(1.3)
        g.Update()
    np.testing.assert_allclose(g[key], 2.75, rtol=0, atol=3e-5)


@pytest.mark.parametrize("method", ["Diffusion", "DiffusionADI"])
def test_periodic_diffusion_sine_mode_converges_to_analytic_decay(api, method):
    # u_t = D u_xx on a periodic domain. Refining dx and dt should reduce
    # error against the exact sinusoidal eigenmode decay.
    errors = []
    D, final_time = 0.35, 0.08
    for n in (32, 64):
        dx = 1.0 / n
        x = np.arange(n, dtype=np.float64) * dx
        initial = np.sin(2*np.pi*x).astype(np.float32)
        # Scale dt with dx^2 for both methods so temporal error cannot mask
        # the expected second-order spatial convergence under refinement.
        dt_target = (0.18 if method == "Diffusion" else 0.5) * dx*dx / D
        steps = int(np.ceil(final_time / dt_target))
        dt = final_time / steps
        g = api.NewPDEgrid((-n,))
        g[:] = initial
        g.SetTimeSpaceStep(dt, dx)
        for _ in range(steps):
            getattr(g, method)(D)
            g.Update()
        exact = np.exp(-D*(2*np.pi)**2*final_time) * np.sin(2*np.pi*x)
        errors.append(float(np.sqrt(np.mean((g[:].astype(np.float64)-exact)**2))))
    assert errors[1] < 0.45 * errors[0]


@pytest.mark.parametrize("method,weight_fn", [
    ("DiffusionRadialCircle", lambda i: i + 0.5),
    ("DiffusionRadialSphere", lambda i: i*i + i + 1/3),
])
def test_radial_zero_flux_conserves_volume_weighted_mass(api, method, weight_fn):
    n = 40
    rng = np.random.default_rng(7890)
    initial = rng.random(n, dtype=np.float32)
    weights = weight_fn(np.arange(n, dtype=np.float64))
    g = api.NewPDEgrid((n,))
    g[:] = initial
    g.SetTimeSpaceStep(0.05, 1.0)
    before = float(np.dot(weights, initial.astype(np.float64)))
    for _ in range(30):
        getattr(g, method)(0.2)
        g.Update()
    after = float(np.dot(weights, g[:].astype(np.float64)))
    assert after == pytest.approx(before, rel=3e-6, abs=3e-6)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
def test_radial_constant_field_fixed_point(api, method):
    g = api.NewPDEgrid((20,))
    g[:] = 3.5
    g.SetTimeSpaceStep(0.1, 0.8)
    for _ in range(15):
        getattr(g, method)(0.2)
        g.Update()
    np.testing.assert_allclose(g[:], 3.5, rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
def test_safe_radial_rejects_wrapped_grid_transactionally(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode requires nonwrapped 1D radial geometry")
    g = api.NewPDEgrid((-10,))
    initial = np.arange(10, dtype=np.float32)
    g[:] = initial
    with pytest.raises(ValueError):
        getattr(g, method)(0.1)
    g.Update()
    np.testing.assert_array_equal(g[:], initial)
