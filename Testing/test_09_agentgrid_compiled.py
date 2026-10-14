"""Compiled AgentGrid iteration/hood tests for PAL's AST lowering layer."""
import numpy as np
import pytest


def test_compiled_all_iteration_reads_each_agent_once(api):
    g = api.NewAgentGrid((12,), numAgentProps=1)
    agents = [g.NewAgentSQ(i) for i in range(8)]
    for n, a in enumerate(agents):
        g[a, 0] = n + 0.5

    @api.njit
    def total(grid):
        count = 0
        value = 0.0
        for agent in grid.All():
            count += 1
            value += grid[agent, 0]
        return count, value

    count, value = total(g)
    assert count == len(agents)
    assert value == pytest.approx(sum(n + 0.5 for n in range(len(agents))))


def test_compiled_agents_at_matches_known_stack(api):
    g = api.NewAgentGrid((5, 5), isStackable=True)
    expected = [g.NewAgentSQ(2, 3) for _ in range(5)]
    g.NewAgentSQ(1, 1)

    @api.njit
    def collect(grid):
        out = np.empty(16, dtype=np.int32)
        n = 0
        for agent in grid.AgentsAt(2, 3):
            out[n] = agent
            n += 1
        return out[:n]

    assert set(map(int, collect(g))) == set(expected)


def test_compiled_hood_visits_independent_expected_sites(api):
    hood = api.MooreHood(2, True)
    g = api.NewAgentGrid((5, 5), isStackable=False)
    center = (2, 2)
    expected_sites = {(center[0] + dx, center[1] + dy) for dx, dy in hood}
    for xy in expected_sites:
        g.NewAgentSQ(*xy)

    @api.njit
    def count_occupied_neighbors(grid):
        total = 0
        for x, y in grid.Hood(hood, 2, 2):
            total += grid.counts[x, y]
        return total

    assert count_occupied_neighbors(g) == len(expected_sites)


def test_compiled_wrapped_hood_visits_each_wrapped_neighbor(api):
    hood = api.VonNeumannHood(2, True)
    g = api.NewAgentGrid((-3, -4), isStackable=False)
    # Independent modulo reference around corner (0,0).
    expected = {((-1) % 3, 0), (1, 0), (0, (-1) % 4), (0, 1)}
    for xy in expected:
        g.NewAgentSQ(*xy)

    @api.njit
    def count_corner(grid):
        total = 0
        for x, y in grid.Hood(hood, 0, 0):
            total += grid.counts[x, y]
        return total

    assert count_corner(g) == len(expected)


def test_compiled_hood_unroll_agrees_with_loop(api):
    hood = api.MooreHood(2, True)
    g = api.NewAgentGrid((-7, -7), isStackable=True)
    for x, y in [(0, 0), (0, 6), (6, 0), (6, 6), (1, 0), (0, 1)]:
        g.NewAgentSQ(x, y)

    @api.njit
    def loop_count(grid):
        n = 0
        for x, y in grid.Hood(hood, 0, 0, unroll=False):
            n += grid.counts[x, y]
        return n

    @api.njit
    def unrolled_count(grid):
        n = 0
        for x, y in grid.Hood(hood, 0, 0, unroll=True):
            n += grid.counts[x, y]
        return n

    assert loop_count(g) == unrolled_count(g) == 5


def test_compiled_all_snapshot_allows_structural_change(api):
    g = api.NewAgentGrid((6,), isStackable=False)
    for i in range(3):
        g.NewAgentSQ(i)

    @api.njit
    def clear_snapshot(grid):
        for agent in grid.All():
            grid.Dispose(agent)

    clear_snapshot(g)
    assert g.GetPop() == 0


def test_compiled_agents_at_detects_structural_change_in_safe_mode(api, safe_mode):
    # AgentsAt lowering saves the linked-list predecessor before user code.
    g = api.NewAgentGrid((4, 4), isStackable=True)
    for _ in range(6):
        g.NewAgentSQ(2, 2)

    @api.njit
    def clear_site(grid):
        for agent in grid.AgentsAt(2, 2):
            grid.Dispose(agent)

    if safe_mode:
        with pytest.raises(RuntimeError, match="structurally modified"):
            clear_site(g)
    else:
        clear_site(g)
        assert g.GetPop() == 0
        assert g.counts[2, 2] == 0


def test_compiled_agents_at_detects_same_site_move_in_safe_mode(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits direct-iteration mutation checks")
    g = api.NewAgentGrid((4, 4), isStackable=True)
    g.NewAgent(2.2, 2.2)
    g.NewAgent(2.4, 2.4)

    @api.njit
    def move_during_iteration(grid):
        for agent in grid.AgentsAt(2, 2):
            grid.Move(agent, 2.7, 2.7)

    with pytest.raises(RuntimeError, match="structurally modified"):
        move_during_iteration(g)
    assert g.GetPop() == 2
    assert g.counts[2, 2] == 2


def test_compiled_agents_at_allows_property_updates(api):
    g = api.NewAgentGrid((4, 4), numAgentProps=1, isStackable=True)
    agents = [g.NewAgentSQ(2, 2) for _ in range(3)]

    @api.njit
    def update_properties(grid):
        count = 0
        for agent in grid.AgentsAt(2, 2):
            grid[agent, 0] = 1.5
            count += 1
        return count

    assert update_properties(g) == len(agents)
    assert all(g[agent, 0] == pytest.approx(1.5) for agent in agents)
    assert g.counts[2, 2] == len(agents)


def test_compiled_agents_in_radius_detects_same_site_move_in_safe_mode(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits direct-iteration mutation checks")
    g = api.NewAgentGrid((4, 4), isStackable=True)
    g.NewAgent(2.2, 2.2)
    g.NewAgent(2.4, 2.4)

    @api.njit
    def move_during_radius_iteration(grid):
        for agent, dx, dy, dist_sq in grid.AgentsInRadius(1.0, 2.5, 2.5):
            grid.Move(agent, 2.7, 2.7)

    with pytest.raises(RuntimeError, match="structurally modified"):
        move_during_radius_iteration(g)
    assert g.GetPop() == 2
    assert g.counts[2, 2] == 2


def test_compiled_agents_in_radius_allows_property_updates(api):
    g = api.NewAgentGrid((4, 4), numAgentProps=1, isStackable=True)
    agents = [g.NewAgent(2.2, 2.2), g.NewAgent(2.4, 2.4)]

    @api.njit
    def update_properties(grid):
        count = 0
        for agent, dx, dy, dist_sq in grid.AgentsInRadius(1.0, 2.5, 2.5):
            grid[agent, 0] = 2.5
            count += 1
        return count

    assert update_properties(g) == len(agents)
    assert all(g[agent, 0] == pytest.approx(2.5) for agent in agents)
    assert g.counts[2, 2] == len(agents)


def test_compiled_radius_matches_python_wrapped_geometry_and_exclusion(api):
    g = api.NewAgentGrid((-6, -7), isStackable=True)
    agents = [g.NewAgent(x, y) for x, y in ((0.2, 0.3), (5.8, 6.8), (1.5, 0.5), (3.5, 3.5))]
    excluded = agents[0]
    radius = 2.0
    cx, cy = 0.1, 0.1
    expected = {int(a): (float(dx), float(dy), float(ds)) for a, dx, dy, ds in g.AgentsInRadius(radius, cx, cy, exclude=excluded)}

    @api.njit
    def collect(grid, ex):
        ids = np.empty(8, dtype=np.int64)
        values = np.empty((8, 3), dtype=np.float64)
        n = 0
        for agent, dx, dy, ds in grid.AgentsInRadius(radius, cx, cy, exclude=ex):
            ids[n] = agent
            values[n, 0] = dx
            values[n, 1] = dy
            values[n, 2] = ds
            n += 1
        return ids[:n], values[:n]

    ids, values = collect(g, excluded)
    actual = {int(a): tuple(row) for a, row in zip(ids, values)}
    assert set(actual) == set(expected)
    assert excluded not in actual
    assert len(actual) == 2
    for agent, expected_values in expected.items():
        assert actual[agent] == pytest.approx(expected_values, abs=2e-6)


@pytest.mark.parametrize("shape,center,radius", [
    ((-6,), (0.1,), 1.6),
    ((-6, -7, -8), (0.1, 0.2, 0.3), 1.5),
])
def test_compiled_radius_matches_python_in_1d_and_3d(api, shape, center, radius):
    dim = len(shape)
    g = api.NewAgentGrid(shape, isStackable=True)
    sites = [tuple(0.2 for _ in shape), tuple(abs(n) - 0.2 for n in shape), tuple(2.5 for _ in shape)]
    agents = [g.NewAgent(*coords) for coords in sites]
    excluded = agents[0]
    expected = {int(row[0]): tuple(float(v) for v in row[1:]) for row in g.AgentsInRadius(radius, *center, exclude=excluded)}

    if dim == 1:
        @api.njit
        def collect(grid, ex):
            ids = np.empty(8, dtype=np.int64)
            vals = np.empty((8, 1), dtype=np.float64)
            count = 0
            for agent, dx in grid.AgentsInRadius(radius, center[0], exclude=ex):
                ids[count] = agent
                vals[count, 0] = dx
                count += 1
            return ids[:count], vals[:count]
    else:
        @api.njit
        def collect(grid, ex):
            ids = np.empty(8, dtype=np.int64)
            vals = np.empty((8, 4), dtype=np.float64)
            count = 0
            for agent, dx, dy, dz, ds in grid.AgentsInRadius(radius, center[0], center[1], center[2], exclude=ex):
                ids[count] = agent
                vals[count, 0] = dx
                vals[count, 1] = dy
                vals[count, 2] = dz
                vals[count, 3] = ds
                count += 1
            return ids[:count], vals[:count]

    ids, vals = collect(g, excluded)
    actual = {int(agent): tuple(row) for agent, row in zip(ids, vals)}
    assert set(actual) == set(expected)
    assert excluded not in actual
    assert len(actual) == 1
    for agent, expected_values in expected.items():
        assert actual[agent] == pytest.approx(expected_values, abs=2e-6)
