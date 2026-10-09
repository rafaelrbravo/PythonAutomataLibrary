"""Regression matrix for AST-rewritten positional/keyword call equivalence.

PAL historically had transformer bugs that silently dropped keyword arguments.
Every specially reconstructed public call covered here is exercised with named
or mixed arguments inside @pal.njit and compared with ordinary Python semantics.
"""
import numpy as np
import pytest


def test_compiled_grid_toi_keywords(api):
    g = api.NewGrid((4, 5, 6))
    @api.njit
    def f(grid):
        return grid.ToI(x=2, y=3, z=4)
    assert f(g) == g.ToI(2, 3, 4)


def test_compiled_pop_toi_add_keywords(api):
    g = api.NewPopGrid((4, 5), capacity=100)
    @api.njit
    def f(grid):
        i = grid.ToI(x=2, y=3)
        grid.Add(value=7, x=2, y=3)
        grid.Update()
        return i
    assert f(g) == g.ToI(2, 3)
    assert g[2, 3] == 7


def test_compiled_pde_toi_add_and_steps_keywords(api):
    g = api.NewPDEgrid((4, 5))
    @api.njit
    def f(grid):
        grid.SetTimeSpaceStep(dt=0.125, dx=0.7, dy=1.3)
        i = grid.ToI(x=1, y=4)
        grid.Add(value=2.5, x=1, y=4)
        grid.Update()
        return i, grid.Dt(), grid.Dx(), grid.Dy()
    got = f(g)
    assert got == pytest.approx((g.ToI(1, 4), 0.125, 0.7, 1.3))
    assert g[1, 4] == pytest.approx(2.5)


def test_compiled_agent_keyword_lifecycle(api):
    g = api.NewAgentGrid((5, 6), isStackable=False)
    @api.njit
    def f(grid):
        a = grid.NewAgentSQ(x=1, y=2)
        i0 = grid.ToI(x=1, y=2)
        last = grid.LastAgent(x=1, y=2)
        grid.MoveSQ(agent=a, x=3, y=4)
        grid.Move(agent=a, x=3.25, y=4.5)
        return a, i0, last, grid.I(a), grid.X(a), grid.Y(a)
    a, i0, last, i1, x, y = f(g)
    assert last == a
    assert i0 == g.ToI(1, 2)
    assert i1 == g.ToI(3, 4)
    assert (x, y) == pytest.approx((3.25, 4.5), abs=1e-6)


@pytest.mark.parametrize("method", ["Diffusion", "DiffusionADI"])
def test_compiled_diffusion_named_rate_and_bcs(api, method):
    initial = np.array([1., 3., 2., 5., 4.], dtype=np.float32)
    a = api.NewPDEgrid((5,)); b = api.NewPDEgrid((5,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(0.04, 0.8); b.SetTimeSpaceStep(0.04, 0.8)
    if method == "Diffusion":
        @api.njit
        def step(grid):
            grid.Diffusion(rateConstant=0.15, xMinBC=2.0, xMaxBC=-1.0)
            grid.Update()
        step(a); b.Diffusion(rateConstant=0.15, xMinBC=2.0, xMaxBC=-1.0); b.Update()
    else:
        @api.njit
        def step(grid):
            grid.DiffusionADI(rateConstant=0.15, xMinBC=2.0, xMaxBC=-1.0)
            grid.Update()
        step(a); b.DiffusionADI(rateConstant=0.15, xMinBC=2.0, xMaxBC=-1.0); b.Update()
    np.testing.assert_allclose(a[:], b[:], rtol=0, atol=1e-7)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
def test_compiled_radial_named_rate(api, method):
    initial = np.array([2., 1., 4., 3., 0.5, 2.5], dtype=np.float32)
    a = api.NewPDEgrid((6,)); b = api.NewPDEgrid((6,))
    a[:] = initial; b[:] = initial
    a.SetTimeSpaceStep(0.03, 0.9); b.SetTimeSpaceStep(0.03, 0.9)
    if method == "DiffusionRadialCircle":
        @api.njit
        def step(grid):
            grid.DiffusionRadialCircle(rateConstant=0.1)
            grid.Update()
        step(a); b.DiffusionRadialCircle(rateConstant=0.1); b.Update()
    else:
        @api.njit
        def step(grid):
            grid.DiffusionRadialSphere(rateConstant=0.1)
            grid.Update()
        step(a); b.DiffusionRadialSphere(rateConstant=0.1); b.Update()
    np.testing.assert_array_equal(a[:], b[:])


def test_compiled_advection_mixed_arguments(api):
    initial = np.arange(20, dtype=np.float32).reshape(4, 5)
    a = api.NewPDEgrid((4, 5)); b = api.NewPDEgrid((4, 5))
    a[:, :] = initial; b[:, :] = initial
    a.SetTimeSpaceStep(0.05, 1.0, 1.2); b.SetTimeSpaceStep(0.05, 1.0, 1.2)
    @api.njit
    def step(grid):
        grid.Advection(0.2, vy=-0.15, xMinBC=3.0, xMaxBC=4.0, yMinBC=5.0, yMaxBC=6.0)
        grid.Update()
    step(a)
    b.Advection(0.2, vy=-0.15, xMinBC=3.0, xMaxBC=4.0, yMinBC=5.0, yMaxBC=6.0); b.Update()
    np.testing.assert_allclose(a[:, :], b[:, :], rtol=0, atol=1e-7)


def test_compiled_loop_keywords_for_hood_box_agents_and_radius(api):
    hood = api.VonNeumannHood(2, True)
    g = api.NewAgentGrid((-5, -5), isStackable=True)
    center = g.NewAgentSQ(0, 0)
    g.NewAgentSQ(1, 0); g.NewAgentSQ(0, 1); g.NewAgentSQ(4, 0); g.NewAgentSQ(0, 4)

    @api.njit
    def f(grid):
        hood_count = 0
        for x, y in grid.Hood(hood=hood, x=0, y=0):
            hood_count += grid.counts[x, y]
        box_count = 0
        for x, y in grid.Box(x1=-1, x2=2, y1=-1, y2=2):
            box_count += grid.counts[x, y]
        at_count = 0
        for a in grid.AgentsAt(x=0, y=0):
            at_count += 1
        radius_count = 0
        for a, dx, dy, dist_sq in grid.AgentsInRadius(rad=1.1, x=0.5, y=0.5, exclude=center):
            radius_count += 1
        return hood_count, box_count, at_count, radius_count

    hood_count, box_count, at_count, radius_count = f(g)
    assert hood_count == 4
    assert box_count == 5
    assert at_count == 1
    assert radius_count == 4
