"""Comparative AgentGrid.All() microbenchmark (run on main and feature branches).

Measures snapshot construction and read-only full iteration, Python and Numba.
Each measurement uses the same grid and reports median wall-clock time.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
import PythonAutomataLibrary as pal

mode = os.environ.get("PAL_TEST_MODE", "safe")
if mode == "fast":
    pal.FastMode()


def python_sum(g):
    total = 0
    for agent in g.All():
        total += int(agent)
    return total


@pal.njit
def compiled_sum(g):
    total = 0
    for agent in g.All():
        total += agent
    return total


def snapshot(g):
    return g.All()


def measure(fn, repetitions):
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return round(statistics.median(times) / 1e6, 5)


for n in (1000, 10000, 100000):
    g = pal.NewAgentGrid((n + 1,), isStackable=False)
    for i in range(n):
        g.NewAgentSQ(i)
    expected = n * (n - 1) // 2
    assert python_sum(g) == expected
    assert compiled_sum(g) == expected
    reps = 30 if n < 100000 else 12
    results = {
        "branch": os.environ.get("BENCH_REVISION", "unknown"),
        "mode": mode,
        "n": n,
        "snapshot_ms": measure(lambda: snapshot(g), reps),
        "python_iteration_ms": measure(lambda: python_sum(g), reps),
        "compiled_iteration_ms": measure(lambda: compiled_sum(g), reps),
    }
    print("BENCH_RESULT " + json.dumps(results), flush=True)
