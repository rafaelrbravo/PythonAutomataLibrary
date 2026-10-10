# PAL public API coverage inventory — 2026-10-09

This inventory maps the intended public surface to dedicated `Testing/API` coverage and the older algorithmic suite. The authoritative integration gate is the complete `Testing/` tree, which includes all dedicated API tests. The audited implementation was promoted to `main`; the permanent `PAL API Audit` workflow now runs on pushes to `main` and pull requests targeting `main` (and retains the historical audit-branch trigger). Current main run `38029920326` completed from repository source with zero failures/xfails: safe **561 passed**; fast **461 passed, 100 skipped**. The two formerly backend-skipped 2D/3D headless OpenGL rendering/save cases now execute and pass under Mesa/EGL in both modes, and the new OpenGL GIF lifecycle regression passes. Windows Python 3.12/MSVC also passed Persian plus four representative real-example regressions in both modes.

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
| OpenGLDraw / OpenGLWindow | `Testing/test_14_visualization.py`, constructor validation in `test_visualization_api.py` | Verified safe/fast headless 2D/3D primitives, update, blocking save, GIF lifecycle, and constructor validation under Mesa/EGL; interactive visible-window behavior remains display/backend dependent |
| AwaitWindows | `test_visualization_api.py` | Verified deterministic headless multi-window lifecycle |
| ColorScale | `test_colorscale_api.py` | Verified endpoints/clamping/interpolation/compiled parity |

## Remaining limitation

Headless OpenGL 2D/3D rendering, blocking save, and GIF lifecycle are now exercised under Mesa/EGL in the permanent Linux gate. Interactive visible-window placement/input/display lifecycle remains inherently display/backend dependent and is not exercised by headless CI. No other known correctness gap remains in the audited public API surface.
