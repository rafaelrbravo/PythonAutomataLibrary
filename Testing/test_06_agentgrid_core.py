"""AgentGrid lifecycle, lattice geometry, occupancy, movement, properties, stacking."""
import itertools

import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 2)])
def test_agentgrid_geometry_matches_c_order(api, shape):
    g = api.NewAgentGrid(shape)
    assert g.nDims == len(shape)
    assert len(g) == int(np.prod(shape))
    for coordinates in itertools.product(*(range(n) for n in shape)):
        i = int(np.ravel_multi_index(coordinates, shape, order="C"))
        assert g.ToI(*coordinates) == i
        if len(shape) > 1:
            assert g.ItoX(i) == coordinates[0]
            assert g.ItoY(i) == coordinates[1]
        if len(shape) > 2:
            assert g.ItoZ(i) == coordinates[2]


def test_nonstackable_lifecycle_and_occupancy(api):
    g = api.NewAgentGrid((4, 5), numAgentProps=2, isStackable=False)
    a = g.NewAgentSQ(1, 2)
    assert g.GetPop() == 1
    assert g.Alive(a)
    assert g.I(a) == g.ToI(1, 2)
    assert (g.XSQ(a), g.YSQ(a)) == (1, 2)
    assert (g.X(a), g.Y(a)) == pytest.approx((1.5, 2.5))
    assert g.LastAgent(1, 2) == a
    assert g.counts[1, 2] == 1
    assert set(map(int, g.All())) == {a}

    g[a, 0] = 3.25
    g[a, 1] = -2.5
    assert g[a, 0] == pytest.approx(3.25)
    assert g[a, 1] == pytest.approx(-2.5)

    g.MoveSQ(a, 3, 4)
    assert g.counts[1, 2] == 0
    assert g.counts[3, 4] == 1
    assert g.I(a) == g.ToI(3, 4)
    assert (g.XSQ(a), g.YSQ(a)) == (3, 4)

    g.Dispose(a)
    assert g.GetPop() == 0
    assert not g.Alive(a)
    assert g.counts[3, 4] == 0
    assert len(g.All()) == 0


def test_continuous_move_updates_square_and_position(api):
    g = api.NewAgentGrid((6, 7), isStackable=False)
    a = g.NewAgent(1.25, 2.75)
    assert (g.X(a), g.Y(a)) == pytest.approx((1.25, 2.75), abs=1e-6)
    assert (g.XSQ(a), g.YSQ(a)) == (1, 2)
    g.Move(a, 4.9, 5.1)
    assert (g.X(a), g.Y(a)) == pytest.approx((4.9, 5.1), abs=1e-5)
    assert (g.XSQ(a), g.YSQ(a)) == (4, 5)
    assert g.counts[1, 2] == 0
    assert g.counts[4, 5] == 1


def test_stackable_counts_and_last_agent(api):
    g = api.NewAgentGrid((3, 3), isStackable=True)
    agents = [g.NewAgentSQ(1, 1) for _ in range(4)]
    assert g.GetPop() == 4
    assert g.counts[1, 1] == 4
    assert g.LastAgent(1, 1) in agents
    g.Dispose(agents[1])
    assert g.GetPop() == 3
    assert g.counts[1, 1] == 3
    assert set(map(int, g.All())) == set(agents) - {agents[1]}


def test_dispose_then_new_agent_reuses_valid_storage(api):
    g = api.NewAgentGrid((10,), isStackable=False)
    old = [g.NewAgentSQ(i) for i in range(5)]
    g.Dispose(old[2])
    replacement = g.NewAgentSQ(7)
    assert g.GetPop() == 5
    assert g.Alive(replacement)
    assert g.I(replacement) == 7
    assert set(map(int, g.All())) == {old[0], old[1], old[3], old[4], replacement}


@pytest.mark.parametrize("dims,axis,size", [((-5,), "X", 5), ((4, -6), "Y", 6), ((3, 4, -7), "Z", 7)])
def test_wrapped_square_coordinates(api, dims, axis, size):
    g = api.NewAgentGrid(dims)
    fn = getattr(g, "InWrapSQ" + axis)
    for coordinate in (-size - 1, -1, 0, size - 1, size, 2 * size + 1):
        assert fn(coordinate) == coordinate % size


def test_safe_nonstackable_rejects_collision(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires caller to honor nonstackable occupancy precondition")
    g = api.NewAgentGrid((4, 4), isStackable=False)
    a = g.NewAgentSQ(1, 1)
    with pytest.raises(ValueError):
        g.NewAgentSQ(1, 1)
    b = g.NewAgentSQ(2, 2)
    with pytest.raises(ValueError):
        g.MoveSQ(b, 1, 1)
    assert g.Alive(a) and g.Alive(b)
    assert g.counts[1, 1] == 1 and g.counts[2, 2] == 1


def test_safe_rejects_dead_agent_operations(api, safe_mode):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise dead-agent validation")
    g = api.NewAgentGrid((4,), numAgentProps=1)
    a = g.NewAgentSQ(1)
    g.Dispose(a)
    for operation in (
        lambda: g.I(a),
        lambda: g.X(a),
        lambda: g.__getitem__((a, 0)),
        lambda: g.MoveSQ(a, 2),
        lambda: g.Dispose(a),
    ):
        with pytest.raises(ValueError):
            operation()


def test_all_shuffle_preserves_membership(api):
    g = api.NewAgentGrid((20,), isStackable=False)
    agents = [g.NewAgentSQ(i) for i in range(12)]
    expected = set(agents)
    for _ in range(8):
        assert set(map(int, g.All(shuffle=True))) == expected
