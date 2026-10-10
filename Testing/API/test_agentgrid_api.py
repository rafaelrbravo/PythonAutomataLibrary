"""AgentGrid public API dimensional, movement, wrapping, and compiled parity."""
import pytest


@pytest.mark.parametrize("dims,sq,pt", [
    ((8,), (3,), (3.25,)),
    ((6, 7), (2, 4), (2.25, 4.75)),
    ((5, 6, 7), (1, 3, 5), (1.25, 3.5, 5.75)),
])
def test_newagent_and_movesq_dimensional_forms(api, dims, sq, pt):
    g = api.NewAgentGrid(dims)
    a = g.NewAgent(*pt)
    assert tuple([g.XSQ(a)] + ([g.YSQ(a)] if len(dims)>1 else []) + ([g.ZSQ(a)] if len(dims)>2 else [])) == sq
    target = tuple((v + 1) % dims[i] for i, v in enumerate(sq))
    g.MoveSQ(a, *target)
    got = tuple([g.XSQ(a)] + ([g.YSQ(a)] if len(dims)>1 else []) + ([g.ZSQ(a)] if len(dims)>2 else []))
    assert got == target


def test_newagentsq_linear_index_form_is_supported_for_all_dimensions(api):
    for dims in ((5,), (3, 4), (3, 4, 5)):
        g = api.NewAgentGrid(dims)
        a = g.NewAgentSQ(3)
        assert g.I(a) == 3


def test_movesq_linear_index_form_is_supported_for_all_dimensions(api):
    for dims in ((5,), (3, 4), (3, 4, 5)):
        g = api.NewAgentGrid(dims)
        a = g.NewAgentSQ(0)
        g.MoveSQ(a, 1)
        assert g.I(a) == 1


def test_lastagent_empty_and_stack_semantics(api):
    g = api.NewAgentGrid((4, 4), isStackable=True)
    assert g.LastAgent(1, 2) == -1
    agents = [g.NewAgentSQ(1, 2) for _ in range(3)]
    assert g.LastAgent(1, 2) == agents[-1]
    g.Dispose(agents[-1])
    assert g.LastAgent(1, 2) == agents[-2]


def test_continuous_wrap_and_displacement(api):
    g = api.NewAgentGrid((-10,))
    assert g.InWrapX(-0.25) == pytest.approx(9.75)
    assert g.InWrapX(10.25) == pytest.approx(0.25)
    assert g.DispWrapX(9.5, 0.5) == pytest.approx(1.0)
    assert g.DispWrapX(0.5, 9.5) == pytest.approx(-1.0)


def test_compiled_keyword_movement_forms(api):
    g = api.NewAgentGrid((5, 6))

    @api.njit
    def work(grid):
        a = grid.NewAgentSQ(x=1, y=2)
        grid.MoveSQ(a, x=3, y=4)
        grid.Move(a, x=2.25, y=1.75)
        return grid.X(a), grid.Y(a), grid.XSQ(a), grid.YSQ(a)

    x, y, xsq, ysq = work(g)
    assert (x, y) == pytest.approx((2.25, 1.75))
    assert (xsq, ysq) == (2, 1)


def test_all_snapshot_survives_disposal(api):
    g = api.NewAgentGrid((8,))
    agents = [g.NewAgentSQ(i) for i in range(5)]
    snapshot = g.All()
    for a in agents:
        g.Dispose(a)
    assert set(map(int, snapshot)) == set(agents)
    assert len(g.All()) == 0
