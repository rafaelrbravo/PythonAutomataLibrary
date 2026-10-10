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


def test_alive_dispose_and_property_roundtrip(api):
    g = api.NewAgentGrid((6,), numAgentProps=2)
    a = g.NewAgentSQ(2)
    assert g.Alive(a)
    g[a, 0] = 1.25
    g[a, 1] = -3.5
    assert (g[a, 0], g[a, 1]) == pytest.approx((1.25, -3.5))
    g.Dispose(a)
    assert not g.Alive(a)


def test_agentsat_and_counts_agree_for_stack(api):
    g = api.NewAgentGrid((5, 5), isStackable=True)
    made = [g.NewAgentSQ(2, 3) for _ in range(4)]
    assert g.counts[2, 3] == 4
    assert set(map(int, g.AgentsAt(2, 3))) == set(made)


def test_all_shuffle_preserves_membership(api):
    g = api.NewAgentGrid((16,))
    made = [g.NewAgentSQ(i) for i in range(8)]
    api.Seed(42)
    shuffled = g.All(shuffle=True)
    assert set(map(int, shuffled)) == set(made)
    assert len(shuffled) == len(made)


def test_discrete_wrap_and_displacement_all_axes(api):
    g = api.NewAgentGrid((-4, -5, -6))
    assert (g.InWrapSQX(-1), g.InWrapSQY(5), g.InWrapSQZ(7)) == (3, 0, 1)
    assert g.DispWrapX(3.5, 0.5) == pytest.approx(1.0)
    assert g.DispWrapY(4.5, 0.5) == pytest.approx(1.0)
    assert g.DispWrapZ(5.5, 0.5) == pytest.approx(1.0)


def test_safe_dead_agent_operations_rejected(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits agent validity checks")
    g = api.NewAgentGrid((5,), numAgentProps=1)
    a = g.NewAgentSQ(1)
    g.Dispose(a)
    for call in (
        lambda: g.I(a),
        lambda: g.XSQ(a),
        lambda: g.X(a),
        lambda: g.__getitem__((a, 0)),
        lambda: g.__setitem__((a, 0), 1.0),
        lambda: g.MoveSQ(a, 2),
        lambda: g.Dispose(a),
    ):
        with pytest.raises(ValueError):
            call()


def test_safe_unstackable_occupancy_rejected_without_state_change(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits occupancy checks")
    g = api.NewAgentGrid((5,))
    a = g.NewAgentSQ(2)
    with pytest.raises(ValueError):
        g.NewAgentSQ(2)
    assert g.GetPop() == 1 and g.LastAgent(2) == a
    b = g.NewAgentSQ(3)
    with pytest.raises(ValueError):
        g.MoveSQ(b, 2)
    assert g.I(b) == 3


def test_python_agents_in_radius_dimensional_tuples(api):
    for dims, pos, center in [
        ((10,), (9.0,), (0.0,)),
        ((10, 10), (9.0, 2.0), (0.0, 2.0)),
        ((10, 10, 10), (9.0, 2.0, 3.0), (0.0, 2.0, 3.0)),
    ]:
        g = api.NewAgentGrid(tuple(-d for d in dims))
        agent = g.NewAgent(*pos)
        found = list(g.AgentsInRadius(1.5, *center))
        assert len(found) == 1
        assert found[0][0] == agent
        assert found[0][1] == pytest.approx(-1.0)
        if len(dims) > 1:
            assert found[0][-1] == pytest.approx(1.0)
        assert list(g.AgentsInRadius(1.5, *center, exclude=agent)) == []


def test_counts_view_is_read_only_and_slice_is_detached(api):
    g = api.NewAgentGrid((4, 5), isStackable=True)
    g.NewAgentSQ(2, 3)
    g.NewAgentSQ(2, 3)
    counts = g.counts
    assert len(counts) == 20
    assert counts[2, 3] == counts[g.ToI(2, 3)] == 2
    linear = counts[0:len(g)]
    assert linear.shape == (20,)
    assert linear[g.ToI(2, 3)] == 2
    region = counts[:, :]
    assert region.shape == (4, 5)
    assert region[2, 3] == 2
    region[2, 3] = 99
    assert counts[2, 3] == 2
    with pytest.raises(TypeError):
        counts[2, 3] = 7


def test_nonspatial_counts_length_is_undefined(api):
    g = api.NewAgentGrid(())
    with pytest.raises(ValueError, match="spatial"):
        len(g.counts)


def test_agentgrid_python_hood_returns_linear_sites(api):
    grid = api.NewAgentGrid((-3, 4))
    offsets = ((-1, 0), (0, 0), (0, -1), (0, 1))
    assert list(grid.Hood(offsets, 0, 0)) == [grid.ToI(2, 0), grid.ToI(0, 0), grid.ToI(0, 1)]
    assert list(grid.Hood(((0, 0), (3, 0)), 1, 2)) == [grid.ToI(1, 2)] * 2
