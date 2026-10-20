"""Precision contracts for public PDEgrid and AgentGrid dtype support."""
import numpy as np
import pytest
import PythonAutomataLibrary as pal

PRECISIONS = [np.float64, np.float32]
AGENT_PRECISIONS = [np.float64, pytest.param(np.float32, marks=pytest.mark.xfail(strict=True, reason="public float32 AgentGrid API not implemented"))]


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_pdegrid_precision(dtype):
    field = pal.NewPDEgrid((4, 4), dtype=dtype)
    value = 1.0 + 1e-10
    field[1, 1] = value
    assert field[1, 1] == float(dtype(value))
    field.Diffusion(0.1)
    field.Update()
    assert np.isfinite(field[1, 1])


@pytest.mark.parametrize("dtype", AGENT_PRECISIONS)
def test_agentgrid_property_precision(dtype):
    grid = pal.NewAgentGrid((4, 4), numAgentProps=1, dtype=dtype)
    agent = grid.NewAgent(1.0, 1.0)
    value = 1.0 + 1e-10
    grid[agent, 0] = value
    assert grid[agent, 0] == float(dtype(value))


@pytest.mark.parametrize("dtype", AGENT_PRECISIONS)
def test_agentgrid_position_precision(dtype):
    grid = pal.NewAgentGrid((4, 4), dtype=dtype)
    x = 1.0 + 1e-10
    agent = grid.NewAgent(x, 1.0)
    assert grid.X(agent) == float(dtype(x))
