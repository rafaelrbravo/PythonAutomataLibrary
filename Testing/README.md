# PAL systematic validation

This directory is both an **executable test suite** and the **audit record** for the Python Automata Library. All repository paths outside `Testing/` are immutable audit baselines. If a bug is found, preserve the original and put the corrected PAL component in `Testing/`; rerun the affected earlier tests against that corrected implementation before continuing.

## Run the suite

From the repository root, using Python with `numpy`, `numba`, and `pytest` installed and a compatible PAL native library available:

```bash
python -m pip install numpy numba pytest
python -m pytest Testing/ -v
PAL_TEST_MODE=fast python -m pytest Testing/ -v
```

On Windows PowerShell, replace the second command with:

```powershell
$env:PAL_TEST_MODE = "fast"
python -m pytest Testing/ -v
Remove-Item Env:PAL_TEST_MODE
```

**Run safe and fast in separate Python processes.** PAL locks the global mode at first construction. Fast mode intentionally skips some input checks; tests for invalid inputs are therefore safe-only. The suite imports `NativeCore.py` from the checkout directly, rather than assuming an installed package. When corrected components exist in `Testing/`, update the shared import fixture to use them; record the switch here and rerun the affected tests.

The test harness is written for `pytest` and does not alter files outside `Testing/` in Git. Python/Numba may create local runtime caches outside `Testing/` when running against source; these are generated artifacts, not committed changes. If strict filesystem immutability is required, run tests against a disposable checkout.

## Performance benchmark

Run compilation/startup and warm-runtime measurements separately from correctness tests:

```bash
python Testing/benchmark_core.py --mode safe --output Testing/results/perf_safe.json
python Testing/benchmark_core.py --mode fast --output Testing/results/perf_fast.json
```

The benchmark records platform, Python/NumPy/Numba versions, construction time, first compiled-call time, and repeated warm-call distributions for AgentGrid movement, PopGrid update, and 2D PDE diffusion. Compare repeated runs rather than interpreting a single small difference. Result paths are deliberately restricted to `Testing/`.

## Coverage matrix

| Order | Subsystem | Tests | Evidence | Status |
| --- | --- | --- | --- | --- |
| 01 | Neighborhood geometry | Moore, von Neumann, Euclidean radius; dimensions 1–3; exclusions; invalid arguments | Independent enumerated coordinate sets | **Authored; execution pending** |
| 02 | Typed Grid | C-order coordinate transforms; 1–3D indexing; slices; wrapping; dtype limits; safe invalid input | NumPy `ravel_multi_index` and arrays | **Authored; execution pending** |
| 03 | IList | append, indexing, snapshots, shuffle permutation, random membership, clear/reuse, safe validation | Independent Python lists/multisets | **Authored; execution pending** |
| 04 | Multinomial | binomial endpoints/ranges/mean; sequential sample conservation; safe invalid arguments | Mathematical invariants and six-SE mean check | **Authored; execution pending** |
| 05 | RNG and Numba | seed replay, ranges, 32/64-bit RandInt, Python/compiled shared stream, IList stream | Deterministic replay and range invariants | **Authored; execution pending** |
| 06 | AgentGrid core | geometry, lifecycle, occupancy, movement, properties, stacking, wrapping, safe collisions/dead agents | Independent lattice/agent invariants | **Authored; execution pending** |
| 07 | PopGrid core | geometry, buffered adds/update, reset, occupancy list, wrapping, atomic invalid updates, capacity | Independent NumPy integer-state reference | **Authored; execution pending** |
| 08 | PDEgrid core | geometry/storage, buffered update/reset, explicit diffusion reference, zero-flux/wrapped conservation, periodic CFL=1 advection, safe transactional stability/CFL checks | Independent finite-difference reference and conservation laws | **Authored; execution pending** |
| 09 | AgentGrid compiled iteration | All, AgentsAt, Hood, wrapped/unrolled Hood, structural mutation guard, dispose-current traversal | Independent expected site/agent sets and cross-path agreement | **Authored; execution pending** |
| 10 | PDEgrid extended | scalar/face BCs, field/interface/mask diffusion, ADI, radial conservation, analytic convergence | Independent flux references, weighted conservation, analytic sine decay | **Authored; execution pending** |
| 11 | Compiled PopGrid/PDEGrid | AST-lowered state updates, keywords, safe source-line diagnostics | Python-path agreement and diagnostic contract | **Authored; execution pending** |
| 12 | Visualization | Pix geometry/bounds, exact headless PNG orientation/RGB, async-save close flushing, OpenGL 2D/3D headless primitives | Independent image decoding and output existence; environment-aware OpenGL skips | **Authored; execution pending** |
| 13 | Integrated models | deterministic agent turnover, agent secretion + diffusion, PopGrid-driven nutrient consumption, advection-diffusion transport | Population/occupancy invariants and independent mass balances | **Authored; execution pending** |\n| 14 | Spatial advection | AdvectionField/AdvectionInterfaces variable velocity, constant-field equivalence, periodic conservation, safe NaN/CFL transactionality | Independent conservative face-flux reference | **Authored; execution pending** |

## Evidence and interpretation

- **Authored ≠ passed.** The tests were inspected against the repository source but **have not been executed in this environment**. No correctness or performance claim is made from their mere presence.
- Current systematic suite: `conftest.py` plus `test_01_hood_geometry.py` through `test_16_pdegrid_advection_fields.py` (1,722 test/harness source lines at the latest audit; performance benchmark counted separately). The coverage ledger was reconciled against these exact paths; a temporary apparent gap was traced to checking abbreviated filenames rather than the actual `_hood_geometry` / `_core` names.
- Log actual command, platform, Python/Numba versions, source revision, mode, pass/fail/skip counts, and failures when an execution environment is available. Add a dated result file under `Testing/`.
- Statistical smoke tests use loose, documented bounds to avoid flaky CI; they do not by themselves certify RNG quality.
- Preserve minimal reproductions for every discovered bug. For each corrected component, record original path/blob SHA, corrected `Testing/` path, affected tests, and regression reruns.
- Keep compile-time and warm-runtime benchmarks separate from correctness assertions. Repeated measurements are required before attributing a performance difference to a code change.

## Baseline provenance

The suite was initialized against `NativeCore.py` blob `f116e2dc9014ca5693d6a27eca4d1052745886ca` and `pal_native.c` blob `df8a9cba08f47bd7735431e52f5ae0afb143e4a3`. Existing root `test_native.py` is historical and remains untouched. No corrected component has yet been produced.
