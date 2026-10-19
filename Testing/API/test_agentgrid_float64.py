"""Double-precision AgentGrid regression tests."""
import numpy as np
import PythonAutomataLibrary as pal


def test_near_upper_boundary_move():
    grid = pal.NewAgentGrid(dimensions=(100, 100), isStackable=True)
    agent = grid.NewAgent(50.0, 50.0)
    x = 100.0 - 1e-8
    grid.Move(agent, x, 50.0)
    assert grid.X(agent) == x



def test_near_upper_boundary_all_axes():
    for dimensions in ((100,), (100, 100), (100, 100, 100)):
        grid = pal.NewAgentGrid(dimensions=dimensions, isStackable=True)
        agent = grid.NewAgent(*([50.0] * len(dimensions)))
        point = [100.0 - 1e-8] * len(dimensions)
        grid.Move(agent, *point)
        assert grid.X(agent) == point[0]
        if len(dimensions) > 1:
            assert grid.Y(agent) == point[1]
        if len(dimensions) > 2:
            assert grid.Z(agent) == point[2]


def test_compiled_near_upper_boundary_all_axes():
    @pal.njit
    def move_2d(grid, agent):
        grid.Move(agent, 100.0 - 1e-8, 100.0 - 1e-8)

    grid = pal.NewAgentGrid(dimensions=(100, 100), isStackable=True)
    agent = grid.NewAgent(50.0, 50.0)
    move_2d(grid, agent)
    assert grid.X(agent) == 100.0 - 1e-8
    assert grid.Y(agent) == 100.0 - 1e-8


def test_compiled_near_upper_boundary_3d():
    @pal.njit
    def move_3d(grid, agent):
        grid.Move(agent, 100.0 - 1e-8, 100.0 - 1e-8, 100.0 - 1e-8)

    grid = pal.NewAgentGrid(dimensions=(100, 100, 100), isStackable=True)
    agent = grid.NewAgent(50.0, 50.0, 50.0)
    move_3d(grid, agent)
    assert (grid.X(agent), grid.Y(agent), grid.Z(agent)) == (100.0 - 1e-8,) * 3


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


def test_compiled_float64_new_agent_only():
    @pal.njit
    def run(grid):
        return grid.NewAgent(5.0 + 1e-10, 5.0)

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    agent = run(grid)
    assert grid.X(agent) == 5.0 + 1e-10


def test_compiled_float64_read_position_only():
    @pal.njit
    def run(grid, agent):
        return grid.X(agent)

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    agent = grid.NewAgent(5.0 + 1e-10, 5.0)
    assert run(grid, agent) == 5.0 + 1e-10


def test_compiled_float64_move_only():
    @pal.njit
    def run(grid, agent):
        grid.Move(agent, 10.0 - 1e-8, 5.0)

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    agent = grid.NewAgent(5.0, 5.0)
    run(grid, agent)
    assert grid.X(agent) == 10.0 - 1e-8



def test_compiled_float64_create_then_move():
    @pal.njit
    def run(grid):
        agent = grid.NewAgent(5.0 + 1e-10, 5.0)
        grid.Move(agent, 10.0 - 1e-8, 5.0)
        return agent

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    agent = run(grid)
    assert grid.X(agent) == 10.0 - 1e-8


def test_compiled_float64_move_then_read():
    @pal.njit
    def run(grid, agent):
        grid.Move(agent, 10.0 - 1e-8, 5.0)
        return grid.X(agent)

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    agent = grid.NewAgent(5.0 + 1e-10, 5.0)
    assert run(grid, agent) == 10.0 - 1e-8


def test_compiled_float64_position():
    @pal.njit
    def run(grid):
        agent = grid.NewAgent(5.0 + 1e-10, 5.0)
        grid.Move(agent, 10.0 - 1e-8, 5.0)
        return grid.X(agent)

    grid = pal.NewAgentGrid(dimensions=(10, 10), isStackable=True)
    assert run(grid) == 10.0 - 1e-8

