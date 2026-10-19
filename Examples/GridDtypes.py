"""Grid dtype constructor showcase.

PDEGrid and AgentGrid currently require float64. Their float32 native backends
are under development; this example will include them once implemented.
"""
import numpy as np
import PythonAutomataLibrary as pal


def main():
    # Grid supports both floating-point precisions today.
    for dtype in (np.float32, np.float64):
        cells = pal.NewGrid((4, 4), dtype=dtype)
        cells[1, 1] = 1.25
        print("Grid", np.dtype(dtype).name, cells[1, 1])

    # PDEGrid currently uses native float64 fields and deltas.
    field = pal.NewPDEgrid((4, 4), dtype=np.float64)
    field[1, 1] = 1.25
    field.Diffusion(0.1)
    field.Update()
    print("PDEGrid float64", field[1, 1])

    # AgentGrid currently uses native float64 positions and properties.
    agents = pal.NewAgentGrid((4, 4), numAgentProps=1, dtype=np.float64)
    agent = agents.NewAgent(1.25, 1.5)
    agents[agent, 0] = 2.5
    print("AgentGrid float64", agents.X(agent), agents[agent, 0])

    # PopGrid retains its existing integer storage choices.
    for dtype in (np.uint8, np.uint16, np.uint32, np.int64):
        populations = pal.NewPopGrid((4, 4), dtype=dtype)
        populations[1, 1] = 2
        populations.Add(3, 1, 1)
        populations.Update()
        print("PopGrid", np.dtype(dtype).name, populations[1, 1])


if __name__ == "__main__":
    main()
