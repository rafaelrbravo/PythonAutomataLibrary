# PAL public API coverage inventory — 2026-10-09

This inventory maps the intended public surface to dedicated `Testing/API` coverage and the older algorithmic suite. The authoritative integration gate is the complete `Testing/` tree, which includes all dedicated API tests. The audited implementation was promoted to `main`; the permanent `PAL API Audit` workflow now runs on pushes to `main` and pull requests targeting `main` (and retains the historical audit-branch trigger). Main run `38028359559` completed from repository source with zero failures/xfails: safe **558 passed, 2 skipped**; fast **458 passed, 102 skipped**. Windows Python 3.12/MSVC also passed Persian plus four representative real-example regressions in both modes.

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

## Remaining limitation

Full OpenGL rendering/save/GIF lifecycle still depends on an available graphics backend. Constructor validation plus headless Pix/save/GIF/AwaitWindows behavior is covered. No other known correctness gap remains in the audited public API surface.
