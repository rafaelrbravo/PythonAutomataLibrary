"""Compiled PopGrid/PDEGrid calls exercise PAL's AST lowering and diagnostics."""
import numpy as np
import pytest


def test_compiled_popgrid_add_update_and_all(api):
    g = api.NewPopGrid((8,), capacity=100)

    @api.njit
    def step(grid):
        grid.Add(4, 2)
        grid.Add(7, 5)
        grid.Update()
        total = 0
        occupied = 0
        for i in grid.All():
            total += grid[i]
            occupied += 1
        return total, occupied

    assert step(g) == (11, 2)
    assert g[2] == 4 and g[5] == 7


def test_compiled_pde_diffusion_matches_python_call(api):
    initial = np.array([0., 1., 4., 2., 0., 3., 1.], dtype=np.float32)
    a = api.NewPDEgrid((-7,)); b = api.NewPDEgrid((-7,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(0.1, 0.8); b.SetTimeSpaceStep(0.1, 0.8)

    @api.njit
    def step(grid):
        grid.Diffusion(0.3)
        grid.Update()

    step(a)
    b.Diffusion(0.3); b.Update()
    np.testing.assert_array_equal(a[:], b[:])


def test_compiled_pde_keyword_arguments_match_python(api):
    initial = np.array([1., 2., 4., 8., 3.], dtype=np.float32)
    a = api.NewPDEgrid((5,)); b = api.NewPDEgrid((5,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(dt=0.05, dx=0.7)
    b.SetTimeSpaceStep(dt=0.05, dx=0.7)

    @api.njit
    def step(grid):
        grid.Diffusion(rateConstant=0.2, xMinBC=5.0, xMaxBC=-1.0)
        grid.Update()

    step(a)
    b.Diffusion(rateConstant=0.2, xMinBC=5.0, xMaxBC=-1.0); b.Update()
    np.testing.assert_array_equal(a[:], b[:])


def test_compiled_pde_advection_keyword_arguments_match_python(api):
    initial = np.arange(9, dtype=np.float32)
    a = api.NewPDEgrid((-9,)); b = api.NewPDEgrid((-9,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(0.2, 1.0); b.SetTimeSpaceStep(0.2, 1.0)

    @api.njit
    def step(grid):
        grid.Advection(vx=-0.7)
        grid.Update()

    step(a)
    b.Advection(vx=-0.7); b.Update()
    np.testing.assert_array_equal(a[:], b[:])


def test_compiled_safe_popgrid_error_contains_source_line(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits validation diagnostics")
    g = api.NewPopGrid((4,), capacity=10)

    @api.njit
    def invalid(grid):
        grid.Add(-1, 0)
        grid.Update()

    with pytest.raises(ValueError) as exc:
        invalid(g)
    # PAL's AST wrappers are intended to preserve model-code source locations.
    assert "source line" in str(exc.value)


def test_compiled_safe_unstable_diffusion_contains_source_line(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits validation diagnostics")
    g = api.NewPDEgrid((5,))
    g.SetTimeSpaceStep(1.0, 1.0)

    @api.njit
    def invalid(grid):
        grid.Diffusion(0.6)

    with pytest.raises(ValueError) as exc:
        invalid(g)
    assert "source line" in str(exc.value)
