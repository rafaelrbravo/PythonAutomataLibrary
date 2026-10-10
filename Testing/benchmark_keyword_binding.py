"""Fresh-process cold/hot baseline for tiny API keyword-name corrections.

Run from the repository parent with NUMBA_DISABLE_CACHE=1 and PAL_BENCH_MODE
set to safe or fast. Each process reports one cold compile+first-call time and
a repeated hot-call median. The workflow repeats fresh processes so the report
can use medians across independent interpreter/compiler starts.
"""
import json
import os
import statistics
import time

import PythonAutomataLibrary as pal

mode = os.environ.get("PAL_BENCH_MODE", "safe")
if mode == "fast":
    pal.FastMode()

grid = pal.NewGrid((-32, -32, -32), int)
items = pal.NewIList()\npop1 = pal.NewPopGrid((-64,))\npde1 = pal.NewPDEgrid((-64,))\nag1 = pal.NewAgentGrid((-64,))

@pal.njit(cache=False)
def grid_wrap_positional(g, n):
    total = 0
    for i in range(n):
        total += g.InWrapX(i - 16)
        total += g.InWrapY(i - 16)
        total += g.InWrapZ(i - 16)
    return total

@pal.njit(cache=False)
def ilist_append_positional(out, n):
    out.Clear()
    for i in range(n):
        out.Append(i)
    return len(out)

def measure(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    cold = time.perf_counter() - t0
    hot = []
    for _ in range(11):
        t0 = time.perf_counter()
        result = fn(*args)
        hot.append(time.perf_counter() - t0)
    return {"cold_s": cold, "hot_median_s": statistics.median(hot), "result": int(result)}

print(json.dumps({
    "mode": mode,
    "grid_wrap": measure(grid_wrap_positional, grid, 200000),
    "ilist_append": measure(ilist_append_positional, items, 200000),\n    "itox_1d": measure(itox_1d_positional, pop1, pde1, ag1, 200000),
}, sort_keys=True))
