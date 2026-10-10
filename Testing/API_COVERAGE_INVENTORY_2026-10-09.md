# PAL public API coverage inventory — 2026-10-09

This inventory maps the intended public surface to dedicated `Testing/API` coverage and the older algorithmic suite. Dedicated API execution is now verified in GitHub Actions from repository source: safe **267 passed**; fast **205 passed, 62 intentional skips**; zero failures and zero xfails at commit `3be0b19742fdc4189fdb5629309093ad4ead7813`.

| Surface | Dedicated API coverage | Older suite / status | Remaining gap |
|---|---|---|---|
| Surface | Dedicated API coverage | Verified status / remaining limitation |
|---|---|---|
| Global mode / constructors | `test_constructors.py` | Verified safe/fast |
| Hood constructors | `test_hood_constructors.py` | Verified safe/fast |
| Seed / Random / RandInt | `test_rng_api.py` | Verified safe/fast; statistical quality also covered by older suite |
| IList | `test_ilist_api.py` | Verified safe/fast |
| Multinomial | `test_multinomial_api.py` | Verified safe/fast |
| Grid | `test_grid_api.py`, `test_grid_slice_assignment.py` | Verified; annotated whole/mixed slice read/write/augassign fixed and passing |
| Shared geometry | `test_shared_geometry_api.py` | Verified; includes safe 1D `ItoX` regression for Agent/Pop/PDE grids |
| AgentGrid | `test_agentgrid_api.py` | Verified lifecycle/property/AgentsAt/shuffle/wrap/invalid-agent/occupancy coverage |
| PopGrid | `test_popgrid_api.py` | Verified overflow/Reset/keyword/indexing contracts; singleton tuple `(i,)` intentionally outside public 1D scalar contract |
| PDEgrid | `test_pdegrid_api.py` | Verified API validation; numerical correctness primarily exercised by older suite |
| AST transformer / diagnostics | `test_transformer_diagnostics_api.py` | Verified; annotated Grid slice parity now ordinary passing coverage |
| Cross-API composition | `test_cross_api_composition.py` | Verified compiled composed kernels |
| Pix / PixWindow | `test_visualization_api.py` | Verified headless Pix/save/GIF/lifecycle cases |
| OpenGLDraw / OpenGLWindow | constructor validation in `test_visualization_api.py` | Validation verified; full rendering/lifecycle/GIF remains backend-dependent |
| AwaitWindows | `test_visualization_api.py` | Verified deterministic headless multi-window lifecycle |
| ColorScale | `test_colorscale_api.py` | Verified endpoints/clamping/interpolation/compiled parity |

## Remaining completeness work

1. Complete the current exact-head full `Testing/` safe/fast integration run and record totals separately from the dedicated API counts.
2. Run representative examples/workloads against the corrected source, including Persian.py where feasible; Windows-specific behavior still requires Windows evidence.
3. Full OpenGL rendering/save/GIF lifecycle remains dependent on an available graphics backend and should be documented rather than simulated as proof.
4. Keep permanent regressions for accepted source fixes and update the native audit report with current CI evidence.
