"""Pickle round-trip tests for PAL native-backed state.

Snapshots must preserve live native state, including pending delta buffers, rather
than process-local pointers.
"""
import pickle

import numpy as np
import pytest


def test_ilist_pickle_roundtrip(api):
    q = api.NewIList()
    for x in (7, -2, 99, 4):
        q.Append(x)
    restored = pickle.loads(pickle.dumps(q))
    assert list(map(int, restored.All())) == [7, -2, 99, 4]


def test_multinomial_pickle_preserves_remaining_state(api):
    m = api.NewMultinomial()
    api.Seed(1234)
    m.Setup(100)
    m.Sample(0.2)
    restored = pickle.loads(pickle.dumps(m))
    # Endpoint p=1 consumes exactly all remaining trials, exposing whether the
    # hidden nRemaining state survived the snapshot without relying on RNG.
    assert restored.Sample(1.0) == 80


def test_popgrid_pickle_preserves_field_and_pending_deltas(api):
    g = api.NewPopGrid((5,), capacity=100)
    g[1] = 10
    g.Add(7, 1)
    g.Add(3, 4)
    restored = pickle.loads(pickle.dumps(g))
    np.testing.assert_array_equal(restored[:], g[:])
    restored.Update()
    assert restored[1] == 17
    assert restored[4] == 3
    assert restored.GetPop() == 20


def test_pdegrid_pickle_preserves_geometry_steps_field_and_deltas(api):
    g = api.NewPDEgrid((-5, 4))
    values = np.arange(20, dtype=np.float32).reshape(5, 4) / 3
    g[:, :] = values
    g.SetTimeSpaceStep(0.2, 0.7, 1.3)
    g.Add(2.5, 2, 1)
    restored = pickle.loads(pickle.dumps(g))
    assert restored.wrapX and not restored.wrapY
    assert restored.Dt() == pytest.approx(0.2)
    assert restored.Dx() == pytest.approx(0.7)
    assert restored.Dy() == pytest.approx(1.3)
    np.testing.assert_array_equal(restored[:, :], values)
    restored.Update()
    expected = values.copy(); expected[2, 1] += 2.5
    np.testing.assert_allclose(restored[:, :], expected, rtol=0, atol=1e-6)


def test_agentgrid_pickle_preserves_lifecycle_links_and_properties(api):
    g = api.NewAgentGrid((-5, 4), numAgentProps=2, isStackable=True)
    agents = [g.NewAgentSQ(1, 2), g.NewAgentSQ(1, 2), g.NewAgentSQ(3, 1)]
    for n, a in enumerate(agents):
        g[a, 0] = 10 + n
        g[a, 1] = -n
    g.Dispose(agents[1])
    restored = pickle.loads(pickle.dumps(g))
    assert restored.GetPop() == 2
    assert restored[1, 2] == 1
    assert restored[3, 1] == 1
    live = list(map(int, restored.All()))
    assert set(live) == {agents[0], agents[2]}
    for a in live:
        assert restored.Alive(a)
        assert restored[a, 0] == pytest.approx(10 + agents.index(a))
    # A new allocation after restore must keep the free-list/lifecycle coherent.
    new_agent = restored.NewAgentSQ(4, 3)
    assert restored.Alive(new_agent)
    assert restored.GetPop() == 3
