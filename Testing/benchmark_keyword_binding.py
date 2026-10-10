"""Fresh-process cold/hot baselines for small PAL API corrections."""
import json
import os
import statistics
import time

import PythonAutomataLibrary as pal

mode = os.environ.get("PAL_BENCH_MODE", "safe")
if mode == "fast":
    pal.FastMode()

grid = pal.NewGrid((-32, -32, -32), int)
items = pal.NewIList()
pop1 = pal.NewPopGrid((-64, -2))
pde1 = pal.NewPDEgrid((-64, -2))
ag1 = pal.NewAgentGrid((-64, -2))
slice_grid = pal.NewGrid((64, 64), int)

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

@pal.njit(cache=False)
def grid_slice_unannotated(g, n):
    total = 0
    for _ in range(n):
        g[:, :] = 3
        total += g[0, 0]
    return total

@pal.njit(cache=False)
def grid_scalar_annotated(g: pal.Grid, n):
    total = 0
    for i in range(n):
        j = i & 63
        g[j, j] = i
        total += g[j, j]
    return total

@pal.njit(cache=False)
def itox_positional(pop, pde, ag, n):
    total = 0
    for i in range(n):
        j = i & 127
        total += pop.ItoX(j)
        total += pde.ItoX(j)
        total += ag.ItoX(j)
    return total

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
    "ilist_append": measure(ilist_append_positional, items, 200000),
    "itox": measure(itox_positional, pop1, pde1, ag1, 200000),
    "grid_slice": measure(grid_slice_unannotated, slice_grid, 2000),
    "grid_annotated_scalar": measure(grid_scalar_annotated, slice_grid, 200000),
}, sort_keys=True))
