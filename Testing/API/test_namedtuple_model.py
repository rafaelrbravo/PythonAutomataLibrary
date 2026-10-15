"""Verify PAL grids and mutable state can travel together through compiled functions."""
from typing import NamedTuple
import numpy as np
import PythonAutomataLibrary as pal

class Model(NamedTuple):
    cells: pal.AgentGrid
    nutrient: pal.PDEgrid
    rates: np.ndarray


def test_namedtuple_composite_across_jitted_functions(api):
    cells = api.NewAgentGrid((8, 8), isStackable=False)
    cells.NewAgentSQ(3, 3)
    nutrient = api.NewPDEgrid((8, 8))
    nutrient[:] = 1.0
    rates = np.array([0.1, 0.2])
    model = Model(cells=cells, nutrient=nutrient, rates=rates)

    @api.njit(cache=False)
    def consume(state: Model):
        for agent in state.cells.All():
            x, y = state.cells.XSQ(agent), state.cells.YSQ(agent)
            state.nutrient[x, y] -= state.rates[0]

    @api.njit(cache=False)
    def step(state: Model):
        consume(state)
        state.rates[1] += 1.0
        return state.nutrient[3, 3]

    assert abs(step(model) - 0.9) < 1e-6
    assert abs(nutrient[3, 3] - 0.9) < 1e-6
    assert abs(rates[1] - 1.2) < 1e-6
    assert abs(step(model) - 0.8) < 1e-6
