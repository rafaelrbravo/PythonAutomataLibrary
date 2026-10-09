"""Cross-component integration tests representative of scientific PAL workloads.

These are intentionally small deterministic models rather than copies of PAL
examples. Their expected behavior follows independent conservation/population
invariants, so integration is not validated by PAL against itself.
"""
import numpy as np
import pytest


def test_agent_birth_death_lifecycle_over_many_steps(api):
    # Deterministic turnover repeatedly exercises allocate/dispose/reuse,
    # occupancy bookkeeping, movement, properties, and compiled iteration.
    g = api.NewAgentGrid((20,), numAgentProps=1)
    for x in range(0, 20, 2):
        a = g.NewAgentSQ(x)
        g[a, 0] = x

    @api.njit
    def turnover(grid, tick):
        for a in grid.All():
            if grid.Alive(a) and (grid.XSQ(a) + tick) % 5 == 0:
                grid.Dispose(a)
        # Fill the first empty site found.
        for x in range(grid.xDim):
            if grid.GetPop(x) == 0:
                a = grid.NewAgentSQ(x)
                grid[a, 0] = tick
                break

    for tick in range(30):
        turnover(g, tick)
        occupied = [x for x in range(20) if g.GetPop(x)]
        live = list(map(int, g.All()))
        assert len(live) == g.GetPop() == len(occupied)
        assert len(set(live)) == len(live)
        assert all(g.Alive(a) for a in live)
        assert sorted(g.XSQ(a) for a in live) == occupied


def test_agent_secretion_diffusion_coupling_mass_balance(api):
    # Agents secrete a fixed amount into a periodic PDE field; diffusion is
    # conservative, so total chemical mass must equal cumulative secretion.
    agents = api.NewAgentGrid((-12, -10))
    field = api.NewPDEgrid((-12, -10))
    sites = [(1, 1), (4, 7), (8, 3), (10, 9)]
    for x, y in sites:
        agents.NewAgentSQ(x, y)
    field.SetTimeSpaceStep(0.05, 1.0, 1.0)
    secretion = 0.125

    @api.njit
    def step(ag, pde):
        for a in ag.All():
            pde.Add(secretion, ag.XSQ(a), ag.YSQ(a))
        pde.Update()
        pde.Diffusion(0.2)
        pde.Update()

    for tick in range(40):
        step(agents, field)
        expected = (tick + 1) * len(sites) * secretion
        assert float(np.sum(field[:, :], dtype=np.float64)) == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_popgrid_reaction_diffusion_coupling_conserves_cells(api):
    # A population field is immobile while its occupancy drives nutrient
    # consumption. This couples integer PopGrid and floating PDEGrid through
    # compiled spatial indexing without stochastic expectations.
    pop = api.NewPopGrid((-16,), capacity=1000)
    nutrient = api.NewPDEgrid((-16,))
    for x, n in [(2, 3), (7, 5), (12, 2)]:
        pop[x] = n
    nutrient[:] = 1.0
    nutrient.SetTimeSpaceStep(0.02, 1.0)
    initial_cells = pop.GetPop()

    @api.njit
    def step(pg, pde):
        for x in pg.All():
            pde.Add(-0.001 * pg[x], x)
        pde.Update()
        pde.Diffusion(0.15)
        pde.Update()

    previous_mass = float(np.sum(nutrient[:], dtype=np.float64))
    for _ in range(25):
        step(pop, nutrient)
        assert pop.GetPop() == initial_cells
        mass = float(np.sum(nutrient[:], dtype=np.float64))
        assert mass < previous_mass
        previous_mass = mass
    expected_loss = 25 * 0.001 * initial_cells
    assert previous_mass == pytest.approx(16.0 - expected_loss, rel=3e-5, abs=3e-5)


def test_advection_diffusion_periodic_transport_preserves_total_mass(api):
    # Representative transport loop combining both operators each tick.
    n = 64
    x = np.arange(n, dtype=np.float64)
    initial = np.exp(-0.5*((x-20)/4)**2).astype(np.float32)
    g = api.NewPDEgrid((-n,))
    g[:] = initial
    g.SetTimeSpaceStep(0.04, 1.0)
    before = float(np.sum(initial, dtype=np.float64))
    for _ in range(100):
        g.Advection(0.6)
        g.Diffusion(0.1)
        g.Update()
    after = float(np.sum(g[:], dtype=np.float64))
    assert after == pytest.approx(before, rel=2e-5, abs=2e-5)
    assert np.all(np.isfinite(g[:]))
    assert np.min(g[:]) >= -1e-6
