# Python / @pal.njit API parity audit

Scope: Compare public input forms, returned values and types, ordered iteration, mutation, exceptions, and numerical results in safe and fast modes. Run each deterministic case against distinct but identically initialized objects. Distinguish source-inspected findings from executed test results. **Do not alter disputed API semantics until user review.**

## Findings for review

| ID | API | Python | @pal.njit | Status | Proposed decision |
| --- | --- | --- | --- | --- | --- |
| H-01 | `grid.Hood(...)` yielded sites | Formerly linear indices | Coordinate scalar/tuples | Resolved with explicit user approval; source and existing tests updated; runtime verification pending | Coordinates in both modes |
| G-01 | `grid.Box(x1=..., x2=..., y1=..., y2=...)` | Python fallback `_PythonBox(self,*bounds)` rejects keywords | Compiled `_HoodExpander._expand_box` accepts keyword bounds; existing `test_compiled_box_keyword_and_positional_forms_match` checks this | Source-confirmed mismatch, not executed | Decide whether Python should accept named bounds for input parity |
| H-03 | `grid.Hood(hood=..., x=..., y=...)` | Python `_PythonHood(self,hood,*coords)` rejects named coordinate arguments | Compiled AST `_bind_args(call,('hood','x','y','z','unroll'),...)` accepts them | Source-confirmed; not executed | Decide whether Python should support named hood coordinates |
| H-02 | `grid.Hood(hood, x[, y[, z]], unroll=True)` | Keyword rejected by Python fallback `_PythonHood(self,hood,*coords)` | Accepted by AST transformer as compile-time bool literal | Source-confirmed mismatch, not yet executed | Consider accepting `unroll` as a Python no-op; user decision pending |

## Paired test inventory (committed; CI execution underway)

- `test_shared_geometry_api.py::test_python_njit_box_and_hood_coordinate_parity`: Grid/PopGrid/PDEgrid × 1D/2D/3D; exact ordered wrapped Box/Hood results.
- `test_agentgrid_api.py::test_python_njit_agentgrid_hood_and_box_order_parity`: AgentGrid × 1D/2D/3D; exact ordered wrapped results.
- `test_agentgrid_api.py::test_python_njit_agent_lifecycle_state_parity`: AgentGrid × 1D/2D/3D; stacked creation, properties, movement, disposal, population, surviving agents.
- `test_agentgrid_api.py::test_python_njit_unstackable_occupancy_failure_atomic_parity`: safe occupied-site creation/movement; compare exception classes and unchanged agent/site state.
- `test_grid_api.py::test_python_njit_grid_invalid_coordinate_read_atomic_parity`: safe-mode invalid X/Y reads, exception class, unchanged arrays.
- `test_grid_api.py::test_python_njit_grid_invalid_linear_read_atomic_parity`: safe-mode negative/out-of-range reads; compare exception classes and unchanged state.
- `test_grid_api.py::test_python_njit_grid_scalar_mutation_parity`: Grid × 1D/2D/3D; coordinate assignment, linear indexing, full-array equality.
- `test_pdegrid_api.py::test_python_njit_pdegrid_add_update_state_parity`: PDEgrid × 1D/2D/3D; pending floating additions, Update, indexing, final fields.
- `test_ilist_api.py::test_python_njit_ilist_invalid_append_atomic_parity`: safe-mode invalid values; compare exception class and unchanged list contents.
- `test_ilist_api.py::test_python_njit_ilist_mutation_and_copy_parity`: IList Append/Iter/All detached-copy/Clear and final-state parity.
- `test_popgrid_api.py::test_python_njit_popgrid_state_transition_parity`: PopGrid × 1D/2D/3D; pending additions, Update, GetPop, indexing, full arrays.
- `test_shared_geometry_api.py::test_python_box_named_bounds_rejected_pending_review`: documents current Python Box keyword rejection (G-01), no semantics changed.
- `test_shared_geometry_api.py::test_python_hood_unroll_keyword_parity_pending_review`: documents current Python rejection without deciding intended behavior.
- `test_shared_geometry_api.py::test_compiled_hood_unroll_matches_default_sites`: compiled unrolled/default ordered coordinates and Python positional reference (H-02); CI pending.
- Existing `test_colorscale_api.py::test_colorscale_compiled_matches_python` and RNG shared-stream tests cover additional narrow parity contracts.

## Execution status

GitHub Actions workflow `pal-api-audit.yml` executes the full `Testing` suite on Linux (safe and fast) and Windows examples. Run `38068395512` at `40a10ec4` completed: Linux safe 620 passed/5 failed; Linux fast 504 passed/116 skipped/5 failed; both Windows jobs succeeded. All five failures were stale test expectations for the approved Hood coordinate-return contract (two cases) or incorrect expected VonNeumannHood order (three cases). Test-only corrections: `09e6fbdb`, `69e2291a`. Follow-up CI run `38068647059` was still in progress when checked; green status is not yet established. Local container cannot execute PAL because the package is absent and github.com DNS is unavailable.

## Next systematic sweep

1. Execute the committed paired tests in separate safe and fast processes; investigate harness errors before classifying API failures.
2. Add paired tests for AgentGrid construction, lifecycle, movement, property access, iteration snapshots and mutation.
3. Add paired tests for Grid scalar and slice indexing, argument defaults and keywords, and validation exceptions.
4. Compare IList, RNG and Multinomial operations and return types, distinguishing seeded-stream equivalence from stochastic distribution equivalence.
5. Compare PDEgrid diffusion/advection and PopGrid reset, overflow and full state; test visualizer methods where compilation is supported.
6. Consolidate verified mismatches into this report for user decisions; leave implementation untouched except explicitly approved fixes.

- `test_grid_api.py::test_safe_grid_failed_write_python_compiled_state_parity`: safe-mode invalid coordinate write, exception-class and unchanged state comparison; unexecuted.

- `test_shared_geometry_api.py::test_python_hood_named_coordinates_rejected_pending_review`: documents current Python keyword rejection across three grid families (H-03).

- `test_shared_geometry_api.py::test_compiled_hood_named_coordinate_forms_match_positional`: compiled named versus positional Hood coordinates and Python positional reference (H-03); unexecuted.

- `test_shared_geometry_api.py::test_compiled_box_named_bounds_match_positional`: compiled Box keyword versus positional coordinates across Grid/PopGrid/PDEgrid (G-01); unexecuted.

CI execution evidence (2026-10-10): GitHub Actions run 38068395512 on commit 40a10ec4 completed Linux safe 620 passed/5 failed and fast 504 passed/116 skipped/5 failed; both Windows example jobs succeeded. Failures were stale coordinate-vs-index expectations in `Testing/test_20_direct_iteration_geometry.py` (2 parametrizations) and wrong hardcoded VonNeumannHood order in `test_shared_geometry_api.py` (3 parametrizations). Test-only corrections committed as 09e6fbdb and 69e2291a. Await subsequent CI to verify; do not claim green yet.

CI follow-up: run `38068647059` (69e2291a, containing both Hood expectation corrections) has successful Windows safe and fast jobs; Linux safe/fast remained in progress at last check. New H-02 compiled unroll regression committed as `6e8f760d`.

Further CI evidence: run `38068643623` (09e6fbdb) Linux safe/fast each had four failures: three pre-fix VonNeumannHood order expectations plus one 1D Hood expected tuple rather than scalar. Run `38068647059` (69e2291a) Linux fast now has only the 1D scalar-vs-tuple failure; Windows safe/fast successful, Linux safe still running when checked. Corrected 1D expectation in `7c9c31c7`; new CI validation pending.
