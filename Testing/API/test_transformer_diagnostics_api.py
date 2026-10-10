"""Annotated AST-transformer parity and diagnostic edge contracts."""
import numpy as np
import pytest
import PythonAutomataLibrary as pal


def test_annotated_grid_augassign_preserves_value(api):
    g = api.NewGrid((4,), np.int32)
    g[2] = 5

    @api.njit
    def work(grid: pal.Grid):
        grid[2] += 3
        return grid[2]

    assert work(g) == 8
    assert g[2] == 8


def test_annotated_popgrid_augassign_is_immediate_value_write(api):
    g = api.NewPopGrid((4,), capacity=100)
    g[1] = 5

    @api.njit
    def work(grid: pal.PopGrid):
        grid[1] += 3
        return grid[1]

    assert work(g) == 8


def test_annotated_pde_augassign_is_immediate_value_write(api):
    g = api.NewPDEgrid((4,))
    g[1] = 1.5

    @api.njit
    def work(grid: pal.PDEgrid):
        grid[1] += 2.25
        return grid[1]

    assert work(g) == pytest.approx(3.75)


def test_annotated_agent_property_augassign(api):
    g = api.NewAgentGrid((4,), numAgentProps=2)
    a = g.NewAgentSQ(1)
    g[a, 0] = 4.0

    @api.njit
    def work(grid: pal.AgentGrid, agent):
        grid[agent, 0] += 2.5
        return grid[agent, 0]

    assert work(g, a) == pytest.approx(6.5)


def test_annotated_ilist_append_keyword_falls_through_with_python_signature(api):
    q = api.NewIList()

    @api.njit
    def work(out: pal.IList):
        out.Append(value=3)
        return out[0]

    assert work(q) == 3


def test_annotated_multinomial_keyword_methods_match_positional(api):
    a = api.NewMultinomial()
    b = api.NewMultinomial()
    api.Seed(123)

    @api.njit
    def keyword(m: pal.Multinomial):
        m.Setup(n=20)
        return m.Binomial(n=10, p=0.25)

    got = keyword(a)
    api.Seed(123)
    b.Setup(20)
    expected = b.Binomial(10, 0.25)
    assert got == expected


def test_annotated_agent_all_default_and_keyword_shuffle(api):
    g = api.NewAgentGrid((8,))
    for i in range(5):
        g.NewAgentSQ(i)

    @api.njit
    def work(grid: pal.AgentGrid):
        a = grid.All()
        b = grid.All(shuffle=False)
        return a, b

    a, b = work(g)
    assert list(a) == list(b)


def test_annotated_safe_grid_augassign_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits validation diagnostics")
    g = api.NewGrid((4,), np.int32)

    @api.njit
    def invalid(grid: pal.Grid):
        grid[9] += 1

    with pytest.raises((IndexError, ValueError), match="source line"):
        invalid(g)


def test_annotated_safe_agent_counts_read_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits validation diagnostics")
    g = api.NewAgentGrid((4,))

    @api.njit
    def invalid(grid: pal.AgentGrid):
        return grid.counts[9]

    with pytest.raises((IndexError, ValueError), match="source line"):
        invalid(g)
