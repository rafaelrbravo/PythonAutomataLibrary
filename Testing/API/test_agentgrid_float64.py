"""Double-precision AgentGrid regression tests."""
import numpy as np
import PythonAutomataLibrary as pal


def test_near_upper_boundary_move():
    grid = pal.NewAgentGrid(dimensions=(100, 100), isStackable=True)
    agent = grid.NewAgent(50.0, 50.0)
    x = 100.0 - 1e-8
    grid.Move(agent, x, 50.0)
    assert grid.X(agent) == x


def test_float64_agent_property():
    grid = pal.NewAgentGrid(dimensions=(10, 10), numAgentProps=1)
    agent = grid.NewAgent(5.0, 5.0)
    value = 1.0 + 1e-10
    grid[agent, 0] = value
    assert grid[agent, 0] == value


def test_float64_position():
    grid = pal.NewAgentGrid(dimensions=(10, 10))
    x = 5.0 + 1e-10
    agent = grid.NewAgent(x, 5.0)
    assert grid.X(agent) == x


def test_compiled_float64_position():
    @pal.njit
    def run(grid):
        agent = grid.NewAgent(5.0 + 1e-10, 5.0)
        grid.Move(agent, 10.0 - 1e-8, 5.0)
        return grid.X(agent)

    grid = pal.NewAgentGrid(dimensions=(10, 10))
    assert run(grid) == 10.0 - 1e-8
