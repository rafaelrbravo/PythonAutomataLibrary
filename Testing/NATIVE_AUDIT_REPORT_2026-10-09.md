# PAL native correctness audit — 2026-10-09

## Scope and source integrity

The 23 `Testing/test_01_*.py` through `test_23_*.py` modules were fetched from the GitHub repository and individually verified against their Git blob SHA-1 before local native execution. The sole test-only correction was committed to `Testing/test_17_pdegrid_boundaries.py` as `93aeb71a1ad2a489fa23099b0670c6c7a08c1425`. `Testing/conftest.py` was also verified byte-for-byte. The PAL implementation was **not** modified. Native execution used a locally staged reconstruction of PAL Python and C sources plus `libpal_native.so`; this is not proof of a reproducible checkout/build from GitHub.

## Consolidated results

Commands, run in **separate fresh processes**:

```sh
PYTHONPATH=/mnt/data/pal_audit_runtime PAL_TEST_MODE=safe python -m pytest /mnt/data/Testing -q --tb=short
PYTHONPATH=/mnt/data/pal_audit_runtime PAL_TEST_MODE=fast python -m pytest /mnt/data/Testing -q --tb=short
```

| Mode | Passed | Failed | Skipped | Xfailed | Warnings | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe | 284 | 2 | 2 | 2 | 4 | 57.45 s |
| Fast | 248 | 0 | 42 | 0 | 3 | 27.93 s |

These are **observed results** on the staged native runtime, not CI results. The safe-mode run has a nonzero exit status due to the two known failures; fast mode exited successfully. No additional full-suite failures were observed.

## Independently rebuilt native library (additional evidence)

A separate local directory was populated with the staged `NativeCore.py`, `OpenGLWindow.py`, `PixWindow.py`, `__init__.py`, and `pal_native.c` source files. The preexisting shared library was **not** copied. The C core was rebuilt using:

```sh
gcc -std=c11 -O3 -fPIC -shared -o libpal_native.so pal_native.c -lm
```

**Import-path correction:** The shared `Testing/conftest.py` inserts its own parent directory into `sys.path`; therefore `PYTHONPATH` alone does not guarantee that a staged test run imports the isolated rebuild. The earlier 15.83 s / 10.46 s runs cannot independently establish which native library was imported. To eliminate this ambiguity, copied `conftest.py` and exact tests 18 (long-horizon PDE) and 21 (state-machine stress) into `/mnt/data/pal_fresh_build/Testing/`, adjacent to the newly rebuilt package, and ran pytest with current working directory `/mnt/data/pal_fresh_build`. **Safe 11/11 passed in 16.35 s; fast 11/11 passed in 9.71 s.** A separate import probe confirmed `PythonAutomataLibrary.__file__` resolves to `/mnt/data/pal_fresh_build/PythonAutomataLibrary/__init__.py`. This confirms the tested behavior with a newly compiled native binary. **It is still not a fresh GitHub checkout:** the Python and C source inputs came from the staged reconstruction, so repository-source provenance and the full fresh-build suite remain unverified.

### Full-suite validation against isolated rebuild

The isolated rebuild was then exercised with **all 23 test modules**, copied into the adjacent `/mnt/data/pal_fresh_build/Testing/` directory together with `conftest.py`. Local byte comparison confirmed all 23 test modules and `conftest.py` matched their previously verified staged counterparts. Running from `/mnt/data/pal_fresh_build` ensures the fixture's `sys.path` insertion selects the rebuilt PAL package.

| Isolated rebuilt binary | Passed | Failed | Skipped | Xfailed | Warnings | Elapsed | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe | 284 | 2 | 2 | 2 | 4 | 56.36 s | 1 |
| Fast | 248 | 0 | 42 | 0 | 3 | 28.70 s | 0 |

The **only** safe failures remain the two known source-line diagnostic assertions in test12; no new failures appeared. The newly compiled `libpal_native.so` SHA-256 `5a444c11149100e694ba307301b4df004e780331553b18be84b897cdefa25968` matched the earlier staged binary. This establishes full-suite behavior on a reproducibly rebuilt binary with import-path isolation, **not** independent retrieval of all implementation sources from a fresh GitHub checkout.

## Known defects and limitations

1. **Compiled safe-mode diagnostic annotation** (unresolved): `test_12_compiled_pop_pde.py` has two failures: `test_compiled_safe_popgrid_error_contains_source_line` and `test_compiled_safe_unstable_diffusion_contains_source_line`. Both raise the expected `ValueError` but the messages omit the asserted `source line` information. `test_23_diagnostic_annotations.py` documents unannotated parameter cases as strict expected failures; annotated cases pass. The observed issue appears related to AST transformer recognition of annotated versus unannotated grid arguments. Do not treat these as numerical-kernel failures. Any implementation fix requires authorization to modify files outside `Testing/`.
2. **Boundary test fixture** (resolved): Original `test_17_pdegrid_boundaries.py` used a one-element list as a dictionary key, causing five `TypeError: unhashable type: 'list'` failures per mode before PAL was invoked. Changed only the test fixture to use the intended string key, committed as `93aeb71`; corrected test: safe 11 passed, fast 10 passed / 1 skipped.
3. **OpenGL**: optional headless OpenGL backend unavailable, so OpenGL cases skip. Headless PixWindow RGB orientation and save tests pass. Visualization emitted Python 3.13 multiprocessing `fork()` deprecation warnings.
4. **Native source provenance**: source files were reconstructed locally from GitHub-accessible content, but the entire native build was not reproduced from a fresh repository checkout in this environment. The test sources were individually hash-verified; this distinction matters.
5. **Statistical tests**: RNG checks are fixed-seed statistical smoke tests, not formal proof of distributional quality. Skipped fast-mode checks generally target deliberate safe-mode validation.

## Representative additional evidence

- 01–07 exact grouped: safe 144 passed; fast 129 passed / 15 skipped.
- 08: safe 22 passed; fast 16 passed / 6 skipped.
- 09: safe 7 passed; fast 7 passed.
- 10: safe 11 passed; fast 8 passed / 3 skipped.
- 11: safe 14 passed; fast 12 passed / 2 skipped.
- 12–13: safe 9 passed / 2 failed; fast 9 passed / 2 skipped.
- 14: safe 5 passed / 2 skipped; fast 3 passed / 4 skipped.
- 15: safe 4 passed; fast 4 passed.
- 16: safe 10 passed; fast 6 passed / 4 skipped.
- 17 (corrected): safe 11 passed; fast 10 passed / 1 skipped.
- 18: safe 8 passed; fast 8 passed.
- 19: safe 9 passed; fast 9 passed.
- 20: safe 15 passed; fast 14 passed / 1 skipped.
- 21: safe 3 passed; fast 3 passed.
- 22: safe 10 passed; fast 10 passed.
- 23: safe 2 passed / 2 strict xfailed; fast 4 skipped.

These module-level results are supplemental, not additive to the consolidated counts.

## Warm execution performance (separate process medians)

Three fresh processes per mode, 30 warm iterations per workload; median of process medians:

| Workload | Safe | Fast | Safe / fast |
| --- | ---: | ---: | ---: |
| AgentGrid 5,000 agents oscillating | 131.375 µs | 29.995 µs | 4.38× |
| PopGrid 10,000 sites Add/Update | 145.3005 µs | 34.0405 µs | 4.27× |
| PDEGrid 256×256 diffusion | 101.1545 µs | 101.315 µs | ~1.00× |

These are local workload measurements, not comprehensive benchmarks or evidence that a particular optimization improved performance. No robust new optimization was established in this audit.

## Diagnostic failure mechanism (source inspection)

Inspection of the staged `NativeCore.py` AST transformer identifies a concrete explanation for the unannotated-argument failures:

- `_HoodExpander.visit_FunctionDef` initializes `self.palVars = {}` and registers parameter names only when their annotations resolve to `Grid`, `AgentGrid`, `PopGrid`, `PDEgrid`, `IList`, or `Multinomial`. An unannotated `grid` parameter is not registered.
- In `_HoodExpander.visit_Call`, the generic method instrumentation appends `_palLine=node.lineno` only if `self.palVars.get(node.func.value.id) not in (None, 'Grid')`. For unannotated parameters, this condition is false, so the call receives no source-line argument.
- The annotated tests in module 23 pass while their unannotated counterparts xfail, independently corroborating this mechanism. The original module 12 tests use unannotated `grid` parameters and reproduce the missing line number.

**Controlled annotation-only reproduction:** Copied the exact six module-12 tests into a temporary local probe file, adding `PopGrid = object` and `PDEgrid = object` marker names and annotating only the two failing `invalid(grid)` parameters as `invalid(grid: PopGrid)` and `invalid(grid: PDEgrid)`. On the same staged native runtime in safe mode, **all 6 tests passed in 9.43 s, exit 0**. The original unannotated test12 still has the two failures. This A/B comparison supports the annotation-gating explanation. The temporary probe was removed; the original test12 and PAL source remain unchanged.

This explains the observed instrumentation gap; it does **not** establish the safest implementation fix. Possible remedies should be evaluated for their effect on ordinary objects with methods sharing PAL names, Numba typing, nested functions, and diagnostics. No library implementation change was made.

### Portable audit runner

The repository now includes `Testing/run_native_audit.py` (commit `559d5cd`) to launch safe and fast suites in independent subprocesses with a per-mode timeout and propagate any nonzero exit status. From a checkout with PAL and pytest installed:

```sh
python Testing/run_native_audit.py
python Testing/run_native_audit.py --mode safe
python Testing/run_native_audit.py --mode fast -- -k diffusion
```

**Runner validation on staged runtime (2026-10-09):** Fetched the committed runner and verified Git blob SHA-1 `a6bec7ecd41f0613e74a6a7881e74bc40933d242` locally. `py_compile` passed. Running `--mode both -- -k test_random_lag1_correlation_smoke` gave safe 1 passed / 289 deselected and fast 1 passed / 289 deselected, overall exit 0. Running `--mode safe -- -k test_compiled_safe_popgrid_error_contains_source_line` reproduced the known diagnostic failure, pytest exit 1 and runner exit 1. Running `--mode safe --timeout 0.001 -- -k test_random_lag1_correlation_smoke` reported timeout status 124 and runner exit 1. These verify success, failure propagation, and timeout propagation in the staged environment; they do not replace fresh-checkout validation.

The runner is a convenience wrapper, **not a test-result artifact**. Its GitHub version has not yet been executed from a fresh checkout; the full-suite results above came from direct pytest commands on the staged runtime. It deliberately returns failure for the currently known two safe-mode diagnostic tests.

## Reproduction and next decisions

For independent reproduction, clone the GitHub PAL repository, build the native C library using the project's supported instructions, install its dependencies, and run `Testing/` in fresh safe/fast processes. The paths above refer to this audit's staging directory, not a portable checkout. Compare the new results with the counts above. Investigate the source-line diagnostic issue with both annotated and unannotated `@njit` grid arguments; preserve exception safety and existing validation semantics. Fixing PAL implementation code requires separate authorization. An OpenGL-capable runner is needed to exercise the skipped graphical backend tests.

**Conclusion:** All 23 test modules were executed with verified test-source identity. Fast-mode suite passes on the staged native runtime; safe-mode suite has exactly two known diagnostic-annotation failures. Numerical and state-machine tests did not uncover additional failures in this run. This is substantial regression evidence, not a formal guarantee of correctness.
