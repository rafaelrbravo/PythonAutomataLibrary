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

    assert loop_count(g) == unrolled_count(g) == 6


def test_compiled_safe_direct_iteration_rejects_structural_change(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits direct-iteration mutation guard")
    g = api.NewAgentGrid((6,), isStackable=False)
    for i in range(3):
        g.NewAgentSQ(i)

    @api.njit
    def invalid(grid):
        for agent in grid.All():
            grid.Dispose(agent)

    with pytest.raises(RuntimeError, match="structurally modified"):
        invalid(g)


def test_compiled_dispose_current_is_supported_in_agents_at(api):
    # AgentsAt lowering saves the linked-list predecessor before user code.
    g = api.NewAgentGrid((4, 4), isStackable=True)
    for _ in range(6):
        g.NewAgentSQ(2, 2)

    @api.njit
    def clear_site(grid):
        for agent in grid.AgentsAt(2, 2):
            grid.Dispose(agent)

    clear_site(g)
    assert g.GetPop() == 0
    assert g[2, 2] == 0
