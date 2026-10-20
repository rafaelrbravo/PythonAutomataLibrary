"""Show PAL grid storage dtypes without changing existing examples.

AgentGrid float32 is available for Python use; compiled Numba parity is pending.
"""
import numpy as np
import PythonAutomataLibrary as pal


def main():
    lattice = pal.NewGrid((4, 4), dtype=np.float64)
    lattice32 = pal.NewGrid((4, 4), dtype=np.float32)
    field64 = pal.NewPDEgrid((4, 4))
    field32 = pal.NewPDEgrid((4, 4), dtype=np.float32)
    agents64 = pal.NewAgentGrid((4, 4), numAgentProps=1)
    agents32 = pal.NewAgentGrid((4, 4), numAgentProps=1, dtype=np.float32)
    counts8 = pal.NewPopGrid((4, 4), dtype=np.uint8)
    counts64 = pal.NewPopGrid((4, 4), dtype=np.int64)

    lattice[1, 2] = 3.5
    lattice32[1, 2] = 3.5
    field64[1, 2] = 2.5
    field32[1, 2] = 2.5
    agent = agents64.NewAgentSQ(1, 2)
    agents64[agent, 0] = 1.25
    agent32 = agents32.NewAgentSQ(1, 2)
    agents32[agent32, 0] = 1.25

    print("Grid float64:", lattice[1, 2])
    print("Grid float32:", lattice32[1, 2])
    print("PDEgrid float64:", field64[1, 2])
    print("PDEgrid float32:", field32[1, 2])
    print("AgentGrid float64 property:", agents64[agent, 0])
    print("AgentGrid float32 property:", agents32[agent32, 0])
    print("PopGrid uint8 and int64:", counts8.xDim, counts64.xDim)


if __name__ == "__main__":
    main()
