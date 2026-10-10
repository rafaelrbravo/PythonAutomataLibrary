# Python / @pal.njit API parity audit

Scope: compare public input forms, returned values/types, ordered iteration, mutation, exceptions, and numerical/state results between ordinary Python and `@pal.njit` in safe and fast modes. Deterministic cases use distinct but identically initialized objects.

## Resolved API discrepancies

| ID | API | Resolution |
| --- | --- | --- |
| H-01 | `grid.Hood(...)` yielded sites | Python now returns coordinate scalar/tuples matching compiled Hood. |
| G-01 | named `grid.Box(x1=..., ...)` bounds | Python and compiled paths both accept public named bounds. |
| H-02 | `grid.Hood(..., unroll=True)` | Compiled path may unroll; Python accepts the keyword as a result-neutral no-op. Yielded sites match. |
| H-03 | named `grid.Hood(hood=..., x=..., ...)` coordinates | Python and compiled paths both accept public named coordinates. |

These changes were explicitly approved during the audit and have permanent regressions.

## Permanent paired coverage

The API suite now compares Python and compiled behavior for:

- Grid, PopGrid, PDEgrid, and AgentGrid Box/Hood ordered coordinates in 1D/2D/3D, including wrapping, duplicate visits, named forms, and Hood unroll.
- Grid scalar mutation, coordinate/linear indexing, ToI mapping, slice read/write behavior, detached slice semantics, and safe invalid read/write exception/state parity.
- AgentGrid construction and deterministic lifecycle behavior: stacked/unstacked creation, movement, properties, disposal, liveness, population, LastAgent behavior, ToI mapping, iteration/snapshot semantics, and safe occupied/dead-agent failures.
- PopGrid Add/Update/Reset, GetPop, scalar/full state, ToI, slice behavior, and safe overflow/validation contracts.
- PDEgrid Add/Update/Reset, scalar/full state, ToI, slice behavior, and dimensional validation.
- IList Append/Clear/reuse, indexing, Iter live storage, All detached snapshots, repeated Clear, and safe invalid Append atomicity.
- PAL RNG stream continuity across Python/compiled call boundaries and RandInt argument contracts.
- Multinomial Setup/Sample/Binomial, including exact reseeded Python/compiled sequences and Setup reuse.
- ColorScale compiled/Python results.
- PDEgrid constant-rate Diffusion and constant-velocity Advection full-field results in 1D/2D/3D.

## Execution evidence

GitHub Actions `pal-api-audit.yml` is the authoritative execution environment. It runs the complete Testing suite on Linux in safe and fast modes and representative Windows safe/fast examples.

Established green checkpoints include run `38068827082` at `7c9c31c7` (Linux safe 637 passed; Linux fast 521 passed, 116 skipped; both Windows jobs successful) and subsequent green extensions through the Grid/PopGrid/PDEgrid/IList/AgentGrid parity additions. Final verification run `38088056182` completed successfully across all four jobs after the Multinomial and PDE Diffusion/Advection additions. Linux safe: 725 passed. Linux fast: 601 passed, 124 skipped. Both Windows safe/fast example jobs succeeded, and the performance benchmark step succeeded in both Linux modes.

## Completion

The planned Python/`@pal.njit` parity audit is complete. No known unresolved public-API semantic mismatch remains. Future paired tests are regression/extension work: if one exposes a mismatch, record the exact input/output/state difference here before changing semantics. Visualization parity remains limited to methods intended to compile; renderer/window lifecycle is covered separately by headless integration tests rather than forced into Python/njit equivalence.

Compiler limitations outside the documented public contract (for example unsupported types beyond a native integer domain) are not parity defects.
