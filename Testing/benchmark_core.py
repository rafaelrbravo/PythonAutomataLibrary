"""Reproducible PAL compilation and warm-runtime benchmark.

Run each mode in a fresh process:
  python Testing/benchmark_core.py --mode safe --output Testing/results/perf_safe.json
  python Testing/benchmark_core.py --mode fast --output Testing/results/perf_fast.json

The benchmark separates construction, first compiled call, and repeated warm
execution. It is descriptive evidence, not a pass/fail correctness test.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NativeCore as pal  # noqa: E402


def timed(fn, repeats=1):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return vals


def summarize(vals):
    return {
        "n": len(vals),
        "median_s": statistics.median(vals),
        "min_s": min(vals),
        "max_s": max(vals),
    }


def bench(name, factory, compiled_step, warm_repeats):
    t0 = time.perf_counter()
    state = factory()
    construct = time.perf_counter() - t0
    first = timed(lambda: compiled_step(*state), 1)[0]
    warm = timed(lambda: compiled_step(*state), warm_repeats)
    return {
        "name": name,
        "construct_s": construct,
        "first_compiled_call_s": first,
        "warm": summarize(warm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("safe", "fast"), required=True)
    ap.add_argument("--warm-repeats", type=int, default=30)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.mode == "fast":
        pal.FastMode()

    @pal.njit
    def agent_step(g):
        for a in g.All():
            x = g.XSQ(a)
            if x + 1 < g.xDim:
                g.MoveSQ(a, x + 1)

    @pal.njit
    def pop_step(g):
        for i in g.All():
            g.Add(1, i)
        g.Update()

    @pal.njit
    def pde_step(g):
        g.Diffusion(0.1)
        g.Update()

    def agent_factory():
        g = pal.NewAgentGrid((10000,))
        for x in range(0, 10000, 2):
            g.NewAgentSQ(x)
        return (g,)

    def pop_factory():
        g = pal.NewPopGrid((10000,), capacity=10**9)
        g[::2] = 1
        return (g,)

    def pde_factory():
        g = pal.NewPDEgrid((-256, -256))
        rng = np.random.default_rng(20261009)
        g[:, :] = rng.random((256, 256), dtype=np.float32)
        g.SetTimeSpaceStep(0.1, 1.0, 1.0)
        return (g,)

    results = {
        "schema": 1,
        "mode": args.mode,
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pal_source": str(ROOT / "NativeCore.py"),
        "warm_repeats": args.warm_repeats,
        "benchmarks": [
            bench("agent_move_1d", agent_factory, agent_step, args.warm_repeats),
            bench("pop_add_update_1d", pop_factory, pop_step, args.warm_repeats),
            bench("pde_diffusion_2d", pde_factory, pde_step, args.warm_repeats),
        ],
    }
    try:
        import numba
        results["numba"] = numba.__version__
    except Exception:
        results["numba"] = None

    encoded = json.dumps(results, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        out = Path(args.output)
        # Audit outputs must stay below Testing/.
        resolved = out.resolve()
        testing = (ROOT / "Testing").resolve()
        if testing not in resolved.parents:
            raise ValueError("--output must be inside Testing/")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
