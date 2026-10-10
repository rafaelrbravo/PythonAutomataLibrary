"""Cross-API compiled composition contracts representative of model kernels."""
import numpy as np
import pytest


def test_agent_query_ilist_popgrid_pde_pipeline(api):
    agents = api.NewAgentGrid((-8, -8), numAgentProps=1)
    selected = api.NewIList()
    pop = api.NewPopGrid((-8, -8), capacity=100)
    field = api.NewPDEgrid((-8, -8))
    hood = api.VonNeumannHood(2, True)
    for x, y in ((0, 0), (1, 0), (7, 0), (0, 7), (4, 4)):
        a = agents.NewAgentSQ(x, y)
        agents[a, 0] = 1.0

    @api.njit
    def step(ag: api.AgentGrid, out: api.IList, pg: api.PopGrid, pde: api.PDEgrid):
        out.Clear()
        for x, y in ag.Hood(hood, 0, 0):
            for a in ag.AgentsAt(x, y):
                out.Append(a)
        for a in out.Iter():
            x = ag.XSQ(a); y = ag.YSQ(a)
            pg.Add(1, x, y)
            pde.Add(ag[a, 0], x, y)
        pg.Update(); pde.Update()
        return len(out), pg.GetPop()

    n, total = step(agents, selected, pop, field)
    assert n == total == 4
    assert float(np.sum(field[:, :], dtype=np.float64)) == pytest.approx(4.0)


def test_rng_multinomial_agentgrid_composition_is_reproducible(api):
    def make():
        return api.NewAgentGrid((32,), isStackable=True), api.NewMultinomial()

    @api.njit
    def seed_population(grid: api.AgentGrid, multi: api.Multinomial):
        multi.Setup(20)
        left = multi.Sample(0.25)
        middle = multi.Sample(0.5)
        right = 20 - left - middle
        for _ in range(left):
            grid.NewAgentSQ(api.RandInt(8))
        for _ in range(middle):
            grid.NewAgentSQ(8 + api.RandInt(8))
        for _ in range(right):
            grid.NewAgentSQ(16 + api.RandInt(16))
        return left, middle, right

    a, ma = make(); b, mb = make()
    api.Seed(20261009)
    ca = seed_population(a, ma)
    occ_a = np.array(a.counts[:], copy=True)
    api.Seed(20261009)
    cb = seed_population(b, mb)
    occ_b = np.array(b.counts[:], copy=True)
    assert ca == cb
    np.testing.assert_array_equal(occ_a, occ_b)
    assert a.GetPop() == b.GetPop() == 20


def test_popgrid_to_pdegrid_to_agentgrid_feedback(api):
    pop = api.NewPopGrid((-10,), capacity=100)
    field = api.NewPDEgrid((-10,))
    agents = api.NewAgentGrid((-10,))
    pop[2] = 3; pop[7] = 2

    @api.njit
    def step(pg: api.PopGrid, pde: api.PDEgrid, ag: api.AgentGrid):
        for x in pg.All():
            pde.Add(0.5 * pg[x], x)
        pde.Update()
        for x in pg.All():
            if pde[x] >= 1.0 and ag.counts[x] == 0:
                ag.NewAgentSQ(x)
        return ag.GetPop()

    assert step(pop, field, agents) == 2
    assert set(agents.XSQ(a) for a in agents.All()) == {2, 7}
    assert field[2] == pytest.approx(1.5)
    assert field[7] == pytest.approx(1.0)


def test_snapshot_query_allows_disposal_and_popgrid_accumulation(api):
    agents = api.NewAgentGrid((12,))
    removed = api.NewPopGrid((12,), capacity=100)
    for x in range(0, 12, 2):
        agents.NewAgentSQ(x)

    @api.njit
    def cull(ag: api.AgentGrid, pg: api.PopGrid):
        snapshot = ag.All()
        for a in snapshot:
            x = ag.XSQ(a)
            pg.Add(1, x)
            ag.Dispose(a)
        pg.Update()
        return ag.GetPop(), pg.GetPop()

    assert cull(agents, removed) == (0, 6)
    assert list(map(int, removed.All())) == list(range(0, 12, 2))
