"""Verify PAL grids and mutable state can travel together through compiled functions."""
from collections import namedtuple

Model = namedtuple("Model", "cells nutrient rates")


def test_namedtuple_composite_across_jitted_functions(api):
    cells = api.NewAgentGrid((8, 8), isStackable=False)
    cells.NewAgentSQ(3, 3)
    nutrient = api.NewPDEgrid((8, 8))
    nutrient[:] = 1.0
    import numpy as np
    rates = np.array([0.1, 0.2])
    model = Model(cells, nutrient, rates)

    @api.njit(cache=False)
    def consume(state):
        for agent in state.cells.All():
            x, y = state.cells.XSQ(agent), state.cells.YSQ(agent)
            state.nutrient[x, y] -= state.rates[0]

    @api.njit(cache=False)
    def step(state):
        consume(state)
        state.rates[1] += 1.0
        return state.nutrient[3, 3]

    assert abs(step(model) - 0.9) < 1e-6
    assert abs(nutrient[3, 3] - 0.9) < 1e-6
    assert abs(rates[1] - 1.2) < 1e-6
    assert abs(step(model) - 0.8) < 1e-6
