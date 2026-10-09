"""Deterministic state-machine stress tests for AgentGrid and PopGrid.

Long randomized operation sequences are checked after every transition against
small independent Python models. Fixed seeds make any failure exactly replayable.
"""
import random

import numpy as np
import pytest


def _agent_snapshot(g):
    live = set(map(int, g.All()))
    by_site = {}
    for a in live:
        site = int(g.I(a))
        by_site.setdefault(site, set()).add(a)
    return live, by_site


def test_stackable_agentgrid_randomized_lifecycle_matches_reference(api):
    rng = random.Random(21001)
    n = 17
    g = api.NewAgentGrid((n,), numAgentProps=1, isStackable=True)
    ref = {}  # agent -> (site, property)
    for step in range(1500):
        op = rng.randrange(4)
        if op == 0 or not ref:
            site = rng.randrange(n)
            a = int(g.NewAgentSQ(site))
            value = rng.uniform(-10, 10)
            g[a, 0] = value
            ref[a] = (site, value)
        elif op == 1:
            a = rng.choice(list(ref))
            site = rng.randrange(n)
            g.MoveSQ(a, site)
            ref[a] = (site, ref[a][1])
        elif op == 2:
            a = rng.choice(list(ref))
            g.Dispose(a)
            del ref[a]
        else:
            a = rng.choice(list(ref))
            value = rng.uniform(-10, 10)
            g[a, 0] = value
            ref[a] = (ref[a][0], value)

        live, by_site = _agent_snapshot(g)
        assert live == set(ref)
        assert g.GetPop() == len(ref)
        for site in range(n):
            assert g.counts[site] == len(by_site.get(site, ()))
        if step % 37 == 0:
            for a, (site, value) in ref.items():
                assert g.I(a) == site
                assert g[a, 0] == pytest.approx(value, abs=1e-6)


def test_nonstackable_agentgrid_randomized_moves_match_reference(api):
    rng = random.Random(21002)
    n = 31
    g = api.NewAgentGrid((n,), isStackable=False)
    ref = {}  # agent -> site
    occupied = {}
    for _ in range(1200):
        empties = [i for i in range(n) if i not in occupied]
        op = rng.randrange(3)
        if (op == 0 and empties) or not ref:
            site = rng.choice(empties)
            a = int(g.NewAgentSQ(site))
            ref[a] = site; occupied[site] = a
        elif op == 1 and empties:
            a = rng.choice(list(ref))
            old = ref[a]; site = rng.choice(empties)
            g.MoveSQ(a, site)
            del occupied[old]; occupied[site] = a; ref[a] = site
        else:
            a = rng.choice(list(ref))
            site = ref.pop(a); del occupied[site]
            g.Dispose(a)

        assert set(map(int, g.All())) == set(ref)
        assert g.GetPop() == len(ref)
        assert [int(g.counts[i]) for i in range(n)] == [int(i in occupied) for i in range(n)]
        for site, a in occupied.items():
            assert g.LastAgent(site) == a


def test_popgrid_randomized_buffered_updates_match_numpy(api):
    rng = random.Random(21003)
    n, capacity = 29, 10_000
    g = api.NewPopGrid((n,), capacity=capacity)
    ref = np.zeros(n, dtype=np.int64)
    for _ in range(300):
        # Generate a valid batch, maintaining nonnegative per-site values and
        # total population below capacity.
        pending = np.zeros(n, dtype=np.int64)
        for _ in range(rng.randrange(1, 20)):
            i = rng.randrange(n)
            lo = -int(ref[i] + pending[i])
            total_after_pending = int(ref.sum() + pending.sum())
            hi = min(20, capacity - total_after_pending)
            if hi < lo:
                continue
            delta = rng.randint(lo, hi)
            g.Add(delta, i)
            pending[i] += delta
        before = g[:].copy()
        np.testing.assert_array_equal(before, ref)
        g.Update()
        ref += pending
        np.testing.assert_array_equal(g[:], ref)
        assert g.GetPop() == int(ref.sum())
        assert set(map(int, g.All())) == set(np.flatnonzero(ref))
