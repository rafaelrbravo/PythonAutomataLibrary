# Python / @pal.njit API parity audit

Scope: Compare public input forms, returned values and types, ordered iteration, mutation, exceptions, and numerical results in safe and fast modes. Run each deterministic case against distinct but identically initialized objects. Distinguish source-inspected findings from executed test results. **Do not alter disputed API semantics until user review.**

## Findings for review

| ID | API | Python | @pal.njit | Status | Proposed decision |
| --- | --- | --- | --- | --- | --- |
| H-01 | `grid.Hood(...)` yielded sites | Formerly linear indices | Coordinate scalar/tuples | Resolved with explicit user approval; CI confirms Python coordinate behavior; full clean suite pending after legacy 1D test correction | Coordinates in both modes |
| G-01 | `grid.Box(x1=..., x2=..., y1=..., y2=...)` | Python fallback `_PythonBox(self,*bounds)` rejects keywords | Compiled `_HoodExpander._expand_box` accepts keyword bounds; existing `test_compiled_box_keyword_and_positional_forms_match` checks this | Source-confirmed; regression tests exercised in green CI | Decide whether Python should accept named bounds for input parity |
| H-03 | `grid.Hood(hood=..., x=..., y=...)` | Python `_PythonHood(self,hood,*coords)` rejects named coordinate arguments | Compiled AST `_bind_args(call,('hood','x','y','z','unroll'),...)` accepts them | Source-confirmed; regression tests exercised in green CI | Decide whether Python should support named hood coordinates |
| H-02 | `grid.Hood(hood, x[, y[, z]], unroll=True)` | Keyword rejected by Python fallback `_PythonHood(self,hood,*coords)` | Accepted by AST transformer as compile-time bool literal | Source-confirmed; regression tests exercised in green CI | Consider accepting `unroll` as a Python no-op; user decision pending |

## Paired test inventory (committed; CI verified)

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

1. Preserve the four-job green baseline from run `38068827082` (`7c9c31c7`); investigate any future regressions against it.
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

CI run `38068743749` (6e8f760d) completed with Windows safe/fast success and exactly one Linux failure in each mode: stale 1D Hood test expected `(x,)` instead of `x`. Thus all newly added compiled named-Box, named-Hood, and unroll tests were included without reported failures in that run, although the overall suite failed. The 1D expectation was fixed in `7c9c31c7`; CI run `38068827082` on that fix was in progress at last check.

CI follow-up: run `38068827082` on 1D Hood test fix `7c9c31c7`: Windows safe/fast completed successfully, Linux safe/fast still in progress when checked. No fresh failures established; do not claim green until both Linux jobs finish.

Post-fix CI result: run `38068827082` on `7c9c31c7` completed Linux fast successfully (**521 passed, 116 skipped**, 91.66 s) and Windows safe/fast successfully. Linux safe remained in progress when checked. This validates the corrected 1D Hood test in fast mode and all newly added compiled keyword/unroll regressions within the passing fast suite; full four-job success remains pending safe completion.

**Four-job green baseline:** GitHub Actions run [`38068827082`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38068827082) on `7c9c31c7` completed SUCCESS. Linux safe **637 passed** in 165.86s; Linux fast **521 passed, 116 skipped** in 91.66s; Windows safe and fast examples both SUCCESS. This executes the committed Python/compiled geometry parity and named-argument/unroll regressions. G-01/H-02/H-03 remain user-review API design discrepancies, not CI failures.

New paired Grid slice coverage: `test_grid_api.py::test_python_njit_grid_slice_read_write_state_parity` exercises 1D/2D/3D stepped slice reads, scalar slice assignments, returned array equality and full final state equality between Python and `@njit`. Committed `e136720f`; CI execution pending. This adds stateful slice parity beyond existing scalar indexing tests; no implementation changes.

Paired PopGrid reset coverage: `test_popgrid_api.py::test_python_njit_popgrid_reset_pending_delta_parity` tests 1D/2D/3D committed population, pending delta, Reset, Update, returned totals and final arrays in Python versus compiled execution. PAL `0bb1d780`; CI pending. Grid slice parity run `38069185811` was still in progress when checked (Windows safe passed). No implementation changes.

Paired PDEgrid reset coverage: `test_pdegrid_api.py::test_python_njit_pdegrid_reset_pending_delta_parity` exercises 1D/2D/3D committed floating field, pending Add, Reset, Update, before/after scalar values and full field equality across Python and compiled execution. PAL `8af77a06`; CI pending. Earlier Grid slice and PopGrid reset Linux CI jobs still in progress at check; no API implementation changes.

Grid slice CI: run `38069185811` on `e136720f` Linux fast SUCCESS, 524 passed/116 skipped in 74.34s, Windows safe/fast SUCCESS; Linux safe in progress at check. Added paired IList Iter-live/All-copy mutation test (`test_ilist_api.py::test_python_njit_ilist_iter_live_and_all_copy_parity`) to compare outputs and final contents in Python and compiled execution, PAL `0bbba550`; new CI pending. PopGrid/PDEgrid reset CI still running.

Verified four-job green extensions: run [`38069185811`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069185811) on `e136720f` (Grid slice parity) Linux safe **640 passed**/167.82s, fast **524 passed, 116 skipped**/74.34s, Windows safe/fast SUCCESS. Run [`38069264438`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069264438) on `0bb1d780` (PopGrid Reset parity) Linux safe **643 passed**/137.57s, fast **527 passed, 116 skipped**/88.18s, Windows safe/fast SUCCESS. PDEgrid Reset and IList live/copy runs remain pending.

Verified PDEgrid reset run [`38069353516`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069353516), PAL `8af77a06`: all four jobs SUCCESS, Linux safe **646 passed**/133.09s, Linux fast **530 passed, 116 skipped**/68.72s, Windows safe/fast SUCCESS. Added paired safe-mode AgentGrid disposed-agent `I` read exception-class and state-preservation test (`test_agentgrid_api.py::test_python_njit_agent_dead_access_failure_atomic_parity`), PAL `c78961a1`; CI pending. IList live/copy parity CI pending. No implementation changes.

Verified IList live/copy CI run [`38069448778`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069448778) on `0bbba550`: all four jobs SUCCESS; Linux safe **647 passed**/141.72s, Linux fast **531 passed, 116 skipped**/88.15s, Windows safe/fast SUCCESS. Added paired AgentGrid All() snapshot surviving disposal test, `test_agentgrid_api.py::test_python_njit_agent_all_snapshot_survives_disposal_parity`, PAL `5a76215c`; CI pending. Prior disposed-agent regression CI pending. No implementation changes.

AgentGrid disposed-agent parity CI progress: run [`38069635005`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069635005) at `c78961a1` Linux fast SUCCESS **531 passed, 117 skipped**/91.35s, Windows safe/fast SUCCESS; Linux safe still running at check. New disposed-agent check is safe-only, so full validation awaits safe job. AgentGrid All snapshot parity run `38069728196` in progress. No implementation changes.

Verified AgentGrid parity runs: [`38069635005`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069635005) on `c78961a1` disposed-agent safe-mode parity, all four jobs SUCCESS: Linux safe **648 passed**/148.21s, Linux fast **531 passed, 117 skipped**/91.35s, Windows safe/fast SUCCESS. [`38069728196`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38069728196) on `5a76215c` All snapshot-disposal parity, all four jobs SUCCESS: Linux safe **649 passed**/107.87s, fast **532 passed, 117 skipped**/51.25s, Windows safe/fast SUCCESS. No implementation changes.

Added paired Grid ToI coordinate mapping parity at first/interior/last sites in 1D/2D/3D, `test_grid_api.py::test_python_njit_grid_to_i_coordinate_parity`, PAL `65c1c381`; CI pending. Latest previously verified test-bearing baseline `5a76215c` four-job green, safe 649 passed, fast 532 passed/117 skipped. No implementation changes.

Added paired IList Clear-and-reuse parity (`test_ilist_api.py::test_python_njit_ilist_clear_and_reuse_parity`), PAL `5bd0d83f`: append two values, record length, clear, append two new values, compare Python/compiled lengths, indexing and complete contents; CI pending. Grid ToI run `38070038829` in progress at last check. No implementation changes.

Added paired PopGrid ToI coordinate mapping regression in 1D/2D/3D, first/interior/last sites (`test_popgrid_api.py::test_python_njit_popgrid_coordinate_mapping_parity`), PAL `0f00bf0d`; CI pending. Grid ToI run `38070038829` Linux fast and Windows safe/fast SUCCESS, Linux safe still in progress at last check. IList Clear/reuse run `38070117718` Windows safe/fast SUCCESS, Linux safe/fast in progress. No implementation changes.

Verified IList Clear/reuse run [`38070117718`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070117718) at `5bd0d83f`: four jobs SUCCESS, Linux safe **653 passed**/110.89s, Linux fast **536 passed, 117 skipped**/78.34s, Windows safe/fast SUCCESS. Added paired PDEgrid ToI coordinate mapping parity at first/interior/last sites in 1D/2D/3D, `test_pdegrid_api.py::test_python_njit_pdegrid_coordinate_mapping_parity`, PAL `20c2bda1`; CI pending. No implementation changes.

Verified Grid ToI run [`38070038829`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070038829) on `65c1c381`, all four jobs SUCCESS: Linux safe **652 passed**/170.96s, fast **535 passed, 117 skipped**/86.55s, Windows safe/fast SUCCESS. Added paired AgentGrid ToI coordinate mapping test (first/interior/last in 1D/2D/3D), `test_agentgrid_api.py::test_python_njit_agentgrid_coordinate_mapping_parity`, PAL `3c3b29e2`; CI pending. No implementation changes.

PopGrid ToI CI [`38070272349`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070272349) on `0f00bf0d` all four jobs SUCCESS (Linux safe **656 passed**, fast **539 passed/117 skipped**, Windows safe/fast success). PDEgrid ToI Linux fast **542 passed/117 skipped** and Windows safe/fast SUCCESS, Linux safe pending; AgentGrid ToI Windows safe/fast SUCCESS, Linux jobs pending. Added paired Grid slice-read aliasing probe (`test_grid_api.py::test_python_njit_grid_slice_read_is_detached_parity`), PAL `5ea86a72`; CI pending. The probe compares Python and compiled returned-slice mutation and source-grid state, without assuming copy versus view semantics. No implementation changes.

PDEgrid ToI CI [`38070330786`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070330786) on `20c2bda1` all four jobs SUCCESS: Linux safe **659 passed**/166.49s, fast **542 passed/117 skipped**/61.48s, Windows safe/fast SUCCESS. AgentGrid ToI Linux fast and Windows safe/fast SUCCESS, Linux safe pending. Added paired PDEgrid slice-read aliasing probe, `test_pdegrid_api.py::test_python_njit_pdegrid_slice_read_aliasing_parity`, PAL `54d0b9a4`; CI pending. This compares returned-slice mutation and source-field state without assuming copy/view behavior. No implementation changes.

AgentGrid ToI CI [`38070418930`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070418930) on `3c3b29e2` all four jobs SUCCESS: Linux safe **662 passed**, fast **545 passed/117 skipped**, Windows safe/fast SUCCESS. This verifies ToI parity coverage across all four grid classes. Added paired PopGrid slice-read aliasing probe (`test_popgrid_api.py::test_python_njit_popgrid_slice_aliasing_parity`), PAL `62a2a406`; CI pending. No implementation changes.

Added paired IList post-Clear index reuse regression (`test_ilist_api.py::test_python_njit_ilist_index_after_clear_parity`), PAL `21ba48f3`, to check stale storage is not exposed after reset and one replacement append; CI pending. Grid/PDEgrid/PopGrid slice aliasing probes remain in-progress in CI. No implementation changes.

PDEgrid slice-read aliasing CI [`38070639735`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070639735) on `54d0b9a4` all four jobs SUCCESS: Linux safe **664 passed**, fast **547 passed/117 skipped**, Windows safe/fast success. Manual safe/fast-mode prose revised per canonical context manual-edit note, `Documentation/MANUAL.md`, PAL `78a9cd53`; no behavior changes. Grid slice aliasing Linux safe still running; PopGrid slice aliasing Linux jobs pending.

Grid slice-read aliasing CI [`38070563446`](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38070563446) on `5ea86a72` four jobs SUCCESS: Linux safe **663 passed**, fast **546 passed/117 skipped**, Windows safe/fast SUCCESS. PopGrid slice aliasing and IList post-Clear indexed reuse Linux fast/Windows SUCCESS, Linux safe pending. Added stacked AgentGrid `LastAgent` before/after disposal Python/njit parity regression, PAL `5ca3fdd9`; CI pending. No implementation changes.

PopGrid slice aliasing CI `38070710248` four jobs SUCCESS: Linux safe **665 passed**, fast **548 passed/117 skipped**, Windows green. IList post-Clear index CI `38070782957` four jobs SUCCESS: Linux safe **666 passed**, fast **549 passed/117 skipped**, Windows green. Added paired stacked AgentGrid `LastAgent` after moving top occupant test, PAL `0cec9429`; CI pending. Prior stacked-disposal parity CI still running. No implementation changes.

Stacked AgentGrid disposal parity run `38070987884` Linux fast FAILED during test construction: test used unsupported `stacking=True` keyword instead of documented `isStackable=True`; this is a test-authoring error, not a PAL parity finding. Corrected both stacked disposal and movement tests in PAL `75d0aa3d`; await corrected CI before claiming either passed. Prior fully green test-bearing baseline remains IList post-Clear run `38070782957`: Linux safe 666 passed, fast 549 passed/117 skipped, Windows green. No PAL implementation changes.

Corrected stacked AgentGrid tests (`isStackable=True`) remain in CI run `38071157733`; no corrected Linux result yet. Added paired IList repeated `Clear()` regression including clearing an empty list and repeated clearing before reuse, PAL `23cd79cd`; CI pending. No PAL implementation changes.
