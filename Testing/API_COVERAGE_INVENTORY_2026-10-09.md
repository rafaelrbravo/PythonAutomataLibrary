# PAL public API coverage inventory — 2026-10-09

This inventory maps the intended public surface to dedicated Testing/API coverage and the older algorithmic suite. It is a coverage checklist, not a pass report: the new API modules have not yet been executed in the current audit runtime.

| Surface | Dedicated API coverage | Older suite / status | Remaining gap |
|---|---|---|---|
| Global mode / constructors | test_constructors.py | conftest.py | Fresh-process execution required |
| Hood constructors | test_hood_constructors.py | test_06_geometry.py | Execute edge contracts |
| Seed / Random / RandInt | test_rng_api.py | test_05_random.py, test_09_random_distributions.py | Execute stream-boundary contracts |
| IList | test_ilist_api.py | test_03_ilist.py, compiled iteration tests | Random() and Shuffle() behavior relies on older suite |
| Multinomial | test_multinomial_api.py | test_09_random_distributions.py | Execute copy/state contracts |
| Grid | test_grid_api.py, test_grid_slice_assignment.py | test_02_grid.py, compiled grid tests | Annotated slice get/set/augassign helper parity; known annotated set xfail |
| Shared geometry | test_shared_geometry_api.py | test_06_geometry.py, test_20_direct_iteration_geometry.py | AgentGrid geometry methods need dedicated parity coverage |
| AgentGrid | test_agentgrid_api.py | test_04_agentgrid.py, state-machine/integration tests | Alive/Dispose, property indexing, AgentsAt and shuffle behavior mainly older-suite covered |
| PopGrid | test_popgrid_api.py | test_07_popgrid_core.py, state-machine tests | Execute overflow/Reset/keyword contracts |
| PDEgrid | test_pdegrid_api.py | test_08/10/16/17 + numerical oracles | Numerical correctness primarily older suite; execute API validation additions |
| AST transformer / diagnostics | test_transformer_diagnostics_api.py | test_22_keyword_transformer_regression.py, test_23_diagnostic_annotations.py | Annotated Grid slice read/set/augassign is main known helper-parity gap |
| Cross-API composition | test_cross_api_composition.py | test_15_integrated_models.py | Execute new composed kernels |
| Pix / PixWindow | test_visualization_api.py | test_14_visualization.py | Execute new compiled/GIF/lifecycle cases |
| OpenGLDraw / OpenGLWindow | constructor validation in test_visualization_api.py | optional real headless primitives/save in test_14_visualization.py | Full method/lifecycle/GIF execution depends on standalone OpenGL backend |
| AwaitWindows | none dedicated | exercised indirectly by examples/window registry behavior | Add deterministic lifecycle test if execution environment supports it |
| ColorScale | none dedicated | example use | Dedicated endpoint/interpolation/validation tests still needed |

## Immediate completeness work

1. Add dedicated ColorScale tests.
2. Expand AgentGrid dedicated tests for Alive, Dispose, property indexing, AgentsAt, All(shuffle=True), discrete wrap/displacement helpers, and safe invalid-agent/occupancy behavior without duplicating state-machine stress tests.
3. Extend annotated Grid regression coverage from whole-slice assignment to slice read and augmented assignment, because transformer helpers _GridGetAt and _GridSetAt are both implicated.
4. Add AwaitWindows lifecycle coverage once a runnable staged package is available; avoid inventing behavior from static inspection.
5. Execute all Testing/API in fresh safe and fast processes before treating this inventory as verified coverage.
6. Only after pre-change cold/hot baselines, implement the minimal annotated Grid slice helper-parity correction and rerun focused/full suites.
