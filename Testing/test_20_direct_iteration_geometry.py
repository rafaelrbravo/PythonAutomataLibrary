"""Direct-iteration geometry and AgentGrid radius semantics.

These exercise PAL's ordinary-Python fallbacks as well as geometry shared with
compiled loop lowering. Expected coordinates/distances are computed independently.
"""
import itertools
import math

import numpy as np
import pytest


@pytest.mark.parametrize("factory", ["NewGrid", "NewPopGrid", "NewPDEgrid"])
def test_python_box_nonwrapped_clips_to_domain(api, factory):
    g = getattr(api, factory)((4, 5), np.float32) if factory == "NewGrid" else getattr(api, factory)((4, 5))
    got = set(tuple(map(int, xy)) for xy in g.Box(-2, 3, 3, 7))
    expected = {(x, y) for x in range(0, 3) for y in range(3, 5)}
    assert got == expected


@pytest.mark.parametrize("factory", ["NewGrid", "NewPopGrid", "NewPDEgrid"])
def test_python_box_wrapped_maps_each_requested_coordinate(api, factory):
    g = getattr(api, factory)((-4, -5), np.float32) if factory == "NewGrid" else getattr(api, factory)((-4, -5))
    got = [tuple(map(int, xy)) for xy in g.Box(-2, 2, 4, 7)]
    expected = [(x % 4, y % 5) for x in range(-2, 2) for y in range(4, 7)]
    assert got == expected


@pytest.mark.parametrize("shape,center", [
    ((-5,), (0,)),
    ((-5, -6), (0, 0)),
    ((-5, -6, -7), (0, 0, 0)),
])
def test_python_hood_matches_independent_modulo_reference(api, shape, center):
    dim = len(shape)
    hood = api.MooreHood(dim, True)
    g = api.NewPopGrid(shape)
    got = list(g.Hood(hood, *center))
    offsets = list(itertools.product((-1, 0, 1), repeat=dim))
    offsets.remove((0,) * dim)
    expected = [tuple((center[d] + off[d]) % abs(shape[d]) for d in range(dim)) for off in offsets]
    assert got == expected


def test_python_agents_at_stack_and_dispose_current(api):
    g = api.NewAgentGrid((4, 4), isStackable=True)
    expected = {g.NewAgentSQ(2, 3) for _ in range(7)}
    g.NewAgentSQ(1, 1)
    seen = set()
    for agent in list(g.AgentsAt(2, 3)):
        seen.add(agent)
        g.Dispose(agent)
    assert seen == expected
    assert g.counts[2, 3] == 0
    assert g.GetPop() == 1


@pytest.mark.parametrize("shape,point,rad", [
    ((-10,), (0.2,), 1.7),
    ((-10, -11), (0.2, 10.7), 2.1),
    ((-10, -11, -12), (9.8, 0.1, 11.9), 2.4),
])
def test_agents_in_radius_matches_minimum_image_geometry(api, shape, point, rad):
    dim = len(shape)
    g = api.NewAgentGrid(shape, isStackable=True)
    coords = list(itertools.product(*(range(abs(n)) for n in shape)))
    agents = []
    for c in coords:
        agents.append((g.NewAgent(*tuple(v + 0.5 for v in c)), c))

    got = {}
    for row in g.AgentsInRadius(rad, *point):
        agent = int(row[0])
        got[agent] = tuple(float(v) for v in row[1:])

    expected_ids = set()
    for agent, c in agents:
        disps = []
        for d, n0 in enumerate(shape):
            n = abs(n0)
            raw = (c[d] + 0.5) - point[d]
            # Same geometric convention independently: shortest signed periodic
            # displacement from query point to agent.
            disp = (raw + n / 2) % n - n / 2
            disps.append(disp)
        dist2 = sum(v * v for v in disps)
        if dist2 <= rad * rad:
            expected_ids.add(agent)
            vals = got[agent]
            assert vals[:dim] == pytest.approx(disps, abs=2e-6)
            if dim >= 2:
                assert vals[-1] == pytest.approx(dist2, abs=3e-6)
    assert set(got) == expected_ids


def test_agents_in_radius_exclude_removes_only_requested_agent(api):
    g = api.NewAgentGrid((-8, -8), isStackable=True)
    agents = [g.NewAgent(3.5, 3.5) for _ in range(4)]
    got = {int(row[0]) for row in g.AgentsInRadius(0.1, 3.5, 3.5, exclude=agents[1])}
    assert got == set(agents) - {agents[1]}


def test_safe_python_direct_all_detects_structural_change(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode deliberately omits generation guard")
    g = api.NewAgentGrid((6,), isStackable=False)
    for i in range(3):
        g.NewAgentSQ(i)
    # All() is a snapshot by contract and therefore permits mutation; direct
    # AgentsAt/AgentsInRadius are the guarded iterators.
    it = g.AgentsInRadius(10.0, 2.5)
    first = next(it)[0]
    g.Dispose(first)
    with pytest.raises(RuntimeError, match="structurally modified"):
        list(it)
