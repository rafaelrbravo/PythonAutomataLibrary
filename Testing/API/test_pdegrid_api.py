"""PDEgrid API state, dimensional validation, and compiled parity."""
import numpy as np
import pytest


@pytest.mark.parametrize("shape,steps", [
    ((5,), (0.2, 0.7)),
    ((4, 5), (0.2, 0.7, 1.1)),
    ((3, 4, 5), (0.2, 0.7, 1.1, 1.4)),
])
def test_time_space_step_accessors(api, shape, steps):
    g = api.NewPDEgrid(shape)
    g.SetTimeSpaceStep(*steps)
    assert g.Dt() == pytest.approx(steps[0])
    assert g.Dx() == pytest.approx(steps[1])
    if len(shape) > 1:
        assert g.Dy() == pytest.approx(steps[2])
    if len(shape) > 2:
        assert g.Dz() == pytest.approx(steps[3])


def test_safe_missing_dimension_accessors_rejected(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits dimensional validation")
    one = api.NewPDEgrid((5,))
    with pytest.raises(ValueError):
        one.Dy()
    with pytest.raises(ValueError):
        one.Dz()
    two = api.NewPDEgrid((4, 5))
    with pytest.raises(ValueError):
        two.Dz()


def test_compiled_reset_clears_field_and_pending_delta(api):
    g = api.NewPDEgrid((6,))
    g[:] = np.arange(6, dtype=np.float32)

    @api.njit
    def work(grid):
        grid.Add(3.5, 2)
        grid.Reset()
        grid.Update()
        return grid[2]

    assert work(g) == pytest.approx(0.0)
    np.testing.assert_array_equal(g[:], np.zeros(6, dtype=np.float32))


def test_compiled_keyword_steps_and_add_3d(api):
    g = api.NewPDEgrid((3, 4, 5))

    @api.njit
    def work(grid):
        grid.SetTimeSpaceStep(dt=0.2, dx=0.7, dy=1.1, dz=1.4)
        grid.Add(value=2.5, x=1, y=2, z=3)
        grid.Update()
        return grid.Dt(), grid.Dx(), grid.Dy(), grid.Dz(), grid[1, 2, 3]

    assert work(g) == pytest.approx((0.2, 0.7, 1.1, 1.4, 2.5))


@pytest.mark.parametrize("method", ["DiffusionMask", "DiffusionField"])
def test_safe_field_size_mismatch_rejected_transactionally(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode assumes correctly sized arrays")
    g = api.NewPDEgrid((4, 5))
    g[:, :] = 1.0
    bad = np.ones(19, dtype=np.float32)
    with pytest.raises(ValueError):
        getattr(g, method)(0.1, bad) if method == "DiffusionMask" else getattr(g, method)(bad)
    g.Update()
    np.testing.assert_array_equal(g[:, :], np.ones((4, 5), dtype=np.float32))


@pytest.mark.parametrize("shape,args", [
    ((5,), (np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32))),
    ((4, 5), (np.ones(20, dtype=np.float32),)),
    ((4, 5), (np.ones(20, dtype=np.float32), np.ones(20, dtype=np.float32), np.ones(20, dtype=np.float32))),
    ((3, 4, 5), (np.ones(60, dtype=np.float32), np.ones(60, dtype=np.float32))),
])
def test_safe_diffusion_interfaces_dimension_arity_rejected(api, safe_mode, shape, args):
    if not safe_mode:
        pytest.skip("Fast mode assumes dimension-appropriate interface arrays")
    g = api.NewPDEgrid(shape)
    with pytest.raises(ValueError):
        g.DiffusionInterfaces(*args)


@pytest.mark.parametrize("method", ["AdvectionField", "AdvectionInterfaces"])
@pytest.mark.parametrize("shape,args", [
    ((5,), (np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32))),
    ((4, 5), (np.zeros(20, dtype=np.float32),)),
    ((4, 5), (np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32))),
    ((3, 4, 5), (np.zeros(60, dtype=np.float32), np.zeros(60, dtype=np.float32))),
])
def test_safe_advection_array_dimension_arity_rejected(api, safe_mode, method, shape, args):
    if not safe_mode:
        pytest.skip("Fast mode assumes dimension-appropriate velocity arrays")
    g = api.NewPDEgrid(shape)
    with pytest.raises(ValueError):
        getattr(g, method)(*args)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
@pytest.mark.parametrize("shape", [(-5,), (3, 4), (1,)])
def test_safe_radial_rejects_invalid_grid_geometry(api, safe_mode, method, shape):
    if not safe_mode:
        pytest.skip("Fast mode assumes radial operator preconditions")
    g = api.NewPDEgrid(shape)
    with pytest.raises(ValueError):
        getattr(g, method)(0.1)


def test_safe_advection_rejects_extra_dimensions(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode assumes dimension-appropriate velocities")
    one = api.NewPDEgrid((5,))
    two = api.NewPDEgrid((4, 5))
    with pytest.raises(ValueError):
        one.Advection(0.1, 0.1)
    with pytest.raises(ValueError):
        two.Advection(0.1, 0.1, 0.1)


@pytest.mark.parametrize("shape", [(7,), (5, 6), (4, 5, 6)])
def test_python_njit_pdegrid_add_update_state_parity(api, shape):
    """Compare deterministic floating-field mutations in both execution modes."""
    py = api.NewPDEgrid(shape)
    jit = api.NewPDEgrid(shape)
    site = tuple(1 for _ in shape)

    def python_work(g):
        g.Add(1.25, *site)
        g.Add(2.5, *site)
        before = g[site if len(shape) > 1 else site[0]]
        g.Update()
        after = g[site if len(shape) > 1 else site[0]]
        g.Add(-0.5, *site)
        g.Update()
        final = g[site if len(shape) > 1 else site[0]]
        return before, after, final

    if len(shape) == 1:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1)
            g.Add(2.5, 1)
            before = g[1]
            g.Update()
            after = g[1]
            g.Add(-0.5, 1)
            g.Update()
            return before, after, g[1]
    elif len(shape) == 2:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1, 1)
            g.Add(2.5, 1, 1)
            before = g[1, 1]
            g.Update()
            after = g[1, 1]
            g.Add(-0.5, 1, 1)
            g.Update()
            return before, after, g[1, 1]
    else:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1, 1, 1)
            g.Add(2.5, 1, 1, 1)
            before = g[1, 1, 1]
            g.Update()
            after = g[1, 1, 1]
            g.Add(-0.5, 1, 1, 1)
            g.Update()
            return before, after, g[1, 1, 1]

    assert compiled_work(jit) == pytest.approx(python_work(py))
    np.testing.assert_allclose(jit[:], py[:], rtol=0, atol=1e-6)


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5)])
def test_python_njit_pdegrid_reset_pending_delta_parity(api, shape):
    """Compare reset after both committed and pending floating-field updates."""
    py = api.NewPDEgrid(shape)
    jit = api.NewPDEgrid(shape)

    def python_work(g):
        g.Add(1.25, *([1] * len(shape)))
        g.Update()
        before = g[tuple([1] * len(shape)) if len(shape) > 1 else 1]
        g.Add(3.5, *([1] * len(shape)))
        g.Reset()
        g.Update()
        return before, g[tuple([1] * len(shape)) if len(shape) > 1 else 1]

    if len(shape) == 1:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1)
            g.Update()
            before = g[1]
            g.Add(3.5, 1)
            g.Reset()
            g.Update()
            return before, g[1]
    elif len(shape) == 2:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1, 1)
            g.Update()
            before = g[1, 1]
            g.Add(3.5, 1, 1)
            g.Reset()
            g.Update()
            return before, g[1, 1]
    else:
        @api.njit
        def compiled_work(g):
            g.Add(1.25, 1, 1, 1)
            g.Update()
            before = g[1, 1, 1]
            g.Add(3.5, 1, 1, 1)
            g.Reset()
            g.Update()
            return before, g[1, 1, 1]

    assert compiled_work(jit) == pytest.approx(python_work(py))
    np.testing.assert_allclose(jit[:], py[:], rtol=0, atol=1e-6)
    assert not np.any(py[:])


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5)])
def test_python_njit_pdegrid_coordinate_mapping_parity(api, shape):
    """Compare ToI mapping at boundary and interior sites."""
    g = api.NewPDEgrid(shape)
    if len(shape) == 1:
        sites = ((0,), (2,), (6,))

        @api.njit
        def compiled(g):
            return g.ToI(0), g.ToI(2), g.ToI(6)
    elif len(shape) == 2:
        sites = ((0, 0), (2, 3), (3, 4))

        @api.njit
        def compiled(g):
            return g.ToI(0, 0), g.ToI(2, 3), g.ToI(3, 4)
    else:
        sites = ((0, 0, 0), (1, 2, 3), (2, 3, 4))

        @api.njit
        def compiled(g):
            return g.ToI(0, 0, 0), g.ToI(1, 2, 3), g.ToI(2, 3, 4)

    expected = tuple(g.ToI(*site) for site in sites)
    assert tuple(compiled(g)) == expected
    assert len(set(expected)) == 3
