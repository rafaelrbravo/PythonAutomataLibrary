# Python / @pal.njit API parity audit

Scope: Compare public input forms, returned values and types, ordered iteration, mutation, exceptions, and numerical results in safe and fast modes. Run each deterministic case against distinct but identically initialized objects. Distinguish source-inspected findings from executed test results. **Do not alter disputed API semantics until user review.**

## Findings for review

| ID | API | Python | @pal.njit | Status | Proposed decision |
| --- | --- | --- | --- | --- | --- |
| H-01 | `grid.Hood(...)` yielded sites | Formerly linear indices | Coordinate scalar/tuples | Resolved with explicit user approval; source and existing tests updated; runtime verification pending | Coordinates in both modes |
| G-01 | `grid.Box(x1=..., x2=..., y1=..., y2=...)` | Python fallback `_PythonBox(self,*bounds)` rejects keywords | Compiled `_HoodExpander._expand_box` accepts keyword bounds; existing `test_compiled_box_keyword_and_positional_forms_match` checks this | Source-confirmed mismatch, not executed | Decide whether Python should accept named bounds for input parity |
| H-02 | `grid.Hood(hood, x[, y[, z]], unroll=True)` | Keyword rejected by Python fallback `_PythonHood(self,hood,*coords)` | Accepted by AST transformer as compile-time bool literal | Source-confirmed mismatch, not yet executed | Consider accepting `unroll` as a Python no-op; user decision pending |

## Paired test inventory (committed; execution pending)

- `test_shared_geometry_api.py::test_python_njit_box_and_hood_coordinate_parity`: Grid/PopGrid/PDEgrid × 1D/2D/3D; exact ordered wrapped Box/Hood results.
- `test_agentgrid_api.py::test_python_njit_agentgrid_hood_and_box_order_parity`: AgentGrid × 1D/2D/3D; exact ordered wrapped results.
- `test_agentgrid_api.py::test_python_njit_agent_lifecycle_state_parity`: AgentGrid × 1D/2D/3D; stacked creation, properties, movement, disposal, population, surviving agents.
- `test_grid_api.py::test_python_njit_grid_scalar_mutation_parity`: Grid × 1D/2D/3D; coordinate assignment, linear indexing, full-array equality.
- `test_pdegrid_api.py::test_python_njit_pdegrid_add_update_state_parity`: PDEgrid × 1D/2D/3D; pending floating additions, Update, indexing, final fields.
- `test_ilist_api.py::test_python_njit_ilist_mutation_and_copy_parity`: IList Append/Iter/All detached-copy/Clear and final-state parity.
- `test_popgrid_api.py::test_python_njit_popgrid_state_transition_parity`: PopGrid × 1D/2D/3D; pending additions, Update, GetPop, indexing, full arrays.
- `test_shared_geometry_api.py::test_python_box_named_bounds_rejected_pending_review`: documents current Python Box keyword rejection (G-01), no semantics changed.
- `test_shared_geometry_api.py::test_python_hood_unroll_keyword_parity_pending_review`: documents current Python rejection without deciding intended behavior.
- Existing `test_colorscale_api.py::test_colorscale_compiled_matches_python` and RNG shared-stream tests cover additional narrow parity contracts.

## Execution status

Not executed in the current audit environment. The Python runtime has NumPy, Numba, and pytest, but no mounted PAL source package; direct GitHub archive access is unavailable from the execution container. GitHub connector source inspection and commits are available, but do not constitute test execution. A clean checkout in CI or a complete local package is required before any passing claim.

## Next systematic sweep

1. Execute the committed paired tests in separate safe and fast processes; investigate harness errors before classifying API failures.
2. Add paired tests for AgentGrid construction, lifecycle, movement, property access, iteration snapshots and mutation.
3. Add paired tests for Grid scalar and slice indexing, argument defaults and keywords, and validation exceptions.
4. Compare IList, RNG and Multinomial operations and return types, distinguishing seeded-stream equivalence from stochastic distribution equivalence.
5. Compare PDEgrid diffusion/advection and PopGrid reset, overflow and full state; test visualizer methods where compilation is supported.
6. Consolidate verified mismatches into this report for user decisions; leave implementation untouched except explicitly approved fixes.
