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
