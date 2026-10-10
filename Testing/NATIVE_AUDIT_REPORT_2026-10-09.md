# PAL correctness and API audit — 2026-10-09

## Current integration baseline

The audited implementation was promoted to `main` by non-forced fast-forward. Final audited implementation/report head: `9e43a36c106eba45b2c8a728aa9dc28c87be1be5`. Post-promotion commit `1fd5f629cac3fb64596b6cabbb64d29f3b4118ec` enabled this audit workflow on future `main` pushes, and `81e5acd95c392dd57c3f2a76857994eee39bf5da` added pull-request validation targeting `main`, preserving the suite as both pre-merge and post-merge regression protection.

GitHub Actions rebuilds the native library from repository source on every run under Python 3.12. The workflow triggers on pushes to `main` and the historical `pal-audit-ci-fix` branch, plus pull requests targeting `main`. Linux uses GCC and runs the complete `Testing/` tree once per safe/fast process. Windows uses MSVC discovered with `vswhere` and runs the real Persian regression plus representative BirthDeath, PopGridExample, ReactionDiffusion2D, and DiffusionAdvection3D smoke tests.

The permanent-main gate has been validated repeatedly after promotion. Run `38028359559` on commit `1fd5f629cac3fb64596b6cabbb64d29f3b4118ec` completed cleanly with zero failures and zero xfails: Linux safe **558 passed, 2 skipped** in 101.34 s; Linux fast **458 passed, 102 skipped** in 39.78 s; Windows safe and fast both built the native DLL with MSVC and passed the five real-source regressions; both Linux performance jobs completed successfully. Subsequent main runs `38028504987` and `38028800566` also completed successfully after documentation/workflow-only maintenance changes.

The prior authoritative clean full-suite baseline before adding the five real-example tests was run `38025875438`: safe **553 passed, 2 skipped**; fast **453 passed, 102 skipped**. The +5 fast count on the current head is exactly the added real-source regression layer.

## Accepted PAL implementation corrections

1. **Compiled Grid wrap keyword binding** — `NativeCore.py` overload parameter names now preserve `InWrapX(x=...)`, `InWrapY(y=...)`, and `InWrapZ(z=...)` in compiled code. Source commit `c567c072`; regression promotion `485c5b49`.
2. **Safe 1D ItoX parity** — AgentGrid, PopGrid, and PDEgrid safe native ItoX implementations now return the linear coordinate in 1D, matching fast/native behavior. Final correction commit `8c785eb4`; focused regression `4b3a7ca`.
3. **Annotated Grid slice assignment/access** — source-aware transformed Grid helpers now delegate slice-containing keys to normal Grid indexing, fixing real code such as Persian's `types[:] = COOPERATOR`. Source commit `46036a26`; promoted regressions `7ddbbf23`.

All three changes were benchmarked before/after in safe and fast modes using fresh-process cache-disabled cold compilation and repeated hot execution. No reproducible runtime regression was accepted.

## Current coverage

Dedicated API coverage now spans constructors/hoods, Grid, AgentGrid, PopGrid, PDEgrid, IList, Multinomial, random, shared geometry, transformer diagnostics, visualization/headless GIF, ColorScale, cross-API composition, and real repository examples. Tests exercise Python vs `@pal.njit`, safe vs fast, positional/keyword/default calls, dimensions, wrapping/boundaries, invalid inputs, invariants, and composed kernels where applicable.

Windows Python 3.12/MSVC now directly validates the original Persian failure family in both safe and fast modes. Full OpenGL window lifecycle remains backend/display dependent; constructor/headless Pix/GIF/AwaitWindows behavior is covered.

## Contract clarifications, not defects

- IList `Append` keyword is `i`, not `value`.
- PopGrid singleton scalar tuple indexing `(0,)` is outside the supported contract.
- `RandInt` is constrained to positive int64 bounds; values beyond int64 are unsupported.
- Hood constructor coercion has some Python-type oddities (for example bool behaving as integer); these are retained behavior, not demonstrated correctness failures.
- AgentGrid linear-index `NewAgentSQ(i)` and `MoveSQ(agent, i)` are deliberately supported in all dimensions.

## CI/cache note

PAL's source transformer creates dynamically compiled Numba functions. Running the real-example tests in two separate pytest invocations in the same workspace can leave/reload artifacts referring to synthetic module `<dynamic>`. This was a CI duplication artifact: `pytest Testing/API` followed by `pytest Testing` ran the same API tests twice. The workflow now uses the complete `pytest Testing` invocation as the single authoritative correctness gate; it already contains all API tests. This preserves coverage without cache-cleanup hacks.

---

## Historical audit record

The material below records earlier staged-source, packaging, and exploratory evidence. Where it conflicts with the current integration baseline above, the current baseline supersedes it.

# PAL native correctness audit — 2026-10-09

## Scope and source integrity

The 23 `Testing/test_01_*.py` through `test_23_*.py` modules were fetched from the GitHub repository and individually verified against their Git blob SHA-1 before local native execution. The sole test-only correction was committed to `Testing/test_17_pdegrid_boundaries.py` as `93aeb71a1ad2a489fa23099b0670c6c7a08c1425`. `Testing/conftest.py` was also verified byte-for-byte. The PAL implementation was **not** modified. Native execution used a locally staged reconstruction of PAL Python and C sources plus `libpal_native.so`; this is not proof of a reproducible checkout/build from GitHub.


## Current GitHub Actions integration evidence

The audit branch now rebuilds `libpal_native.so` directly from repository `pal_native.c` under Python 3.12 and runs both the dedicated API suite and the complete `Testing/` tree in separate safe/fast jobs.

At commit `019a138698e4769273f942c97f85b74e2626c31e`, both jobs completed successfully with zero failures and zero xfails. Safe: dedicated API **267 passed**; complete `Testing/` **553 passed, 2 skipped**. Fast: dedicated API **205 passed, 62 skipped**; complete `Testing/` **453 passed, 102 skipped**.

This supersedes the earlier provenance limitation for current integration evidence: the results come from a GitHub checkout plus a native library freshly compiled in each job, rather than a locally reconstructed source tree. Older staged/rebuilt results below are retained as historical evidence, not the current pass report.

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

## GitHub core-source provenance verified (2026-10-09)

Queried GitHub's recursive tree for default-branch commit `da275e53ad9ef93cc5bd28cb7c74a33ef27d754b` using `GET /repos/rafaelrbravo/PythonAutomataLibrary/git/trees/main?recursive=1` (`truncated=false`). Computed canonical Git blob SHA-1 locally for each file in the isolated rebuild (`sha1(b"blob " + len(bytes) + b"\\0" + bytes)`). **All five PAL core implementation source files matched GitHub byte-for-byte:**

| Root-level source | GitHub tree Git blob SHA-1 | Local isolated build |
| --- | --- | --- |
| `NativeCore.py` | `f116e2dc9014ca5693d6a27eca4d1052745886ca` | match |
| `pal_native.c` | `df8a9cba08f47bd7735431e52f5ae0afb143e4a3` | match |
| `OpenGLWindow.py` | `580fbc9c7c6996dbd09507639f9c675a63347e27` | match |
| `PixWindow.py` | `38889ea0bb50c8f4510142885a4ba29fea99dcc6` | match |
| `__init__.py` | `2399e5d02ec075d36dee86d0e5dd28f2f7363b6a` | match |

These are the **repository-root** source paths, not `PythonAutomataLibrary/NativeCore.py` paths. The isolated build places these files inside a package directory for import. All 23 test modules and `conftest.py` had previously been verified against GitHub blob hashes. Consequently, the full-suite results on the isolated rebuild use **byte-exact GitHub core sources and tests**, not unverified source reconstructions. A literal clean `git clone` and end-to-end repository packaging/install procedure have still not been executed, and optional OpenGL behavior remains untested.

## Distribution packaging failure (new finding)

A separate wheel-build probe reproduced the repository-root `pyproject.toml` exactly in a temporary packaging directory, alongside the five byte-verified PAL core source files and the already rebuilt `libpal_native.so`. Ran:

```sh
python -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
```

The build succeeded, producing `pythonautomatalibrary-0.1.0-py3-none-any.whl` (63,566 bytes). **Zip inspection found only `NativeCore.py`, `OpenGLWindow.py`, `PixWindow.py`, `__init__.py` and `.dist-info` metadata. Neither `libpal_native.so` nor `pal_native.c` was packaged.** A wheel installed from this artifact would not carry the required native library. The universal `py3-none-any` tag is also inappropriate for a wheel that distributes compiled platform-specific code. This is an independently observed packaging defect, not a runtime numerical test failure.

**Installed-wheel import failure confirmed:** Ran `python -m pip install --no-deps --target /mnt/data/pal_wheel_install_probe <built-wheel>` successfully, then started Python outside the repository with `PYTHONPATH` pointing to that target. `import PythonAutomataLibrary` failed immediately in `NativeCore.py` at the `ctypes.CDLL` load with `OSError: .../PythonAutomataLibrary/libpal_native.so: cannot open shared object file: No such file or directory`. This demonstrates an end-user import failure, not merely incomplete archive metadata. Added reusable `Testing/smoke_wheel_install.py` (commit `4609a88`), which installs a supplied wheel into a temporary target, launches a separate Python process outside the repository, verifies the imported module originates in the target, and propagates import failure. Running this smoke test against the defective wheel returned exit 1 with the same missing-library `OSError`. The test is standalone and does not change library code.

Added standalone `Testing/check_wheel_contents.py` (commit `06bff34`) to verify that a given built wheel contains a native shared library and is not mislabeled as universal. This check intentionally fails for the observed wheel; it is not automatically collected by pytest. Packaging fixes would require authorization to edit root-level `pyproject.toml` and possibly the build backend integration, which is outside the present `Testing/`-only scope.

### Temporary packaging-fix experiment (not applied to PAL)

A separate copy of the GitHub-verified source package was modified **outside the repository** for a controlled packaging experiment. Appended the following to its `pyproject.toml`:

```toml
[tool.setuptools.package-data]
PythonAutomataLibrary = ["libpal_native.so", "libpal_native.dylib", "pal_native.dll"]
```

Created this `setup.py` alongside it:

```python
from setuptools import setup, Distribution

class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(distclass=NativeDistribution)
```

Built using `python -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .`. The new `pythonautomatalibrary-0.1.0-cp313-cp313-linux_x86_64.whl` contained `PythonAutomataLibrary/libpal_native.so`. The original defective universal wheel remained in the same output directory from an earlier build, so **the platform-specific wheel was selected explicitly** for validation. Ran `Testing/smoke_wheel_install.py` against that wheel: pip installation succeeded and the fresh subprocess imported PAL from the isolated target, exit 0.

This is a working **Linux/Python 3.13 proof of concept**, not a committed packaging fix or a verified cross-platform build process. A robust production solution should integrate compilation with the wheel build (rather than requiring a prebuilt `.so`), choose appropriate wheel compatibility tags, and test Windows/macOS packaging. The workaround intentionally remains outside the permitted `Testing/` modifications.

### Build-time native compilation proof of concept (Linux)

Extended the temporary packaging experiment to start **without** a prebuilt `libpal_native.so`: copied the source-only packaging probe to `/mnt/data/pal_packaging_build_probe`, removed `libpal_native.so`, `build/`, `dist/`, and egg-info, then defined a temporary `setuptools.command.build_py.build_py` subclass. Its `run()` calls `super().run()` and then compiles the repository's `pal_native.c` into `Path(self.build_lib)/"PythonAutomataLibrary"/"libpal_native.so"` with `cc -O3 -shared -fPIC -std=c11 ... -lm`. Kept the `Distribution.has_ext_modules() -> True` override and native package-data declaration from the previous experiment.

The clean `python -m pip wheel --no-build-isolation --no-deps -w dist .` completed successfully, generating `pythonautomatalibrary-0.1.0-cp313-cp313-linux_x86_64.whl` (136,102 bytes; SHA256 `e435902ade06092c8794814fc28f9f416837dba57686cb81dc1c7af56e0c4281`). ZIP inspection verified `PythonAutomataLibrary/libpal_native.so` is present. Running `Testing/smoke_wheel_install.py` against this wheel succeeded, with the isolated subprocess importing PAL from its temporary installation.

**Caveats:** this is a Linux/Python 3.13 prototype, not an authorized change to root packaging. The source-root `setup.py` was also copied into the wheel due to the current root-level package-directory mapping; production packaging should exclude it. Windows/MSVC, macOS, clean build isolation, source distribution, and CI testing remain open. This experiment confirms a feasible native compilation integration, not complete release readiness.

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

## Final complete rerun after diagnostic test annotations (2026-10-09)

After the user-requested change in `Testing/test_12_compiled_pop_pde.py` (commit `6df2c02ad7adad4779a4b6b7e315dc9f320a7279`), updated the corresponding two test functions in the isolated build's test tree with `grid: PopGrid` and `grid: PDEgrid` marker annotations. No PAL implementation files were modified. Executed the **entire 23-module suite** in separate fresh Python processes from `/mnt/data/pal_fresh_build`, against the previously GitHub-blob-verified source files and rebuilt native library:

| Mode | Passed | Failed | Skipped | Xfailed | Warnings | Time | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe | **286** | **0** | 2 | 2 | 4 | 57.08 s | 0 |
| Fast | **248** | **0** | 42 | 0 | 3 | 27.73 s | 0 |

The two formerly failing source-line assertions now pass with annotations. Two strict expected failures in `test_23_diagnostic_annotations.py` continue to characterize the unsupported source-line enhancement for **unannotated** grid arguments. Warnings are Python 3.13 multiprocessing `fork()` deprecations in visualization tests. The wheel packaging defect remains a separate, confirmed distribution problem; the green suite does not resolve it.


## Post-promotion headless OpenGL closure (2026-10-10)

The permanent Linux gate was extended with ModernGL and Mesa/EGL so the existing display-free OpenGL regressions execute instead of skipping. This exposed a real safe-mode defect: `_OpenGLWindowSafe._Save(block=True)` called a missing `_WaitAck`, and the concrete safe class also lacked `StartGif`, `AddGifFrame`, and `StopGif`. Commit `0e3dc5aa82296432a6c53b1883866e0653a1025a` restores the acknowledgement helper and validated safe GIF API; commit `ef2c9854f08493dd0f857295db15024f3067f029` adds a headless OpenGL GIF lifecycle regression.

Main workflow run `38029920326` is green in all four jobs. Linux results are **561 passed** in safe mode and **461 passed, 100 skipped** in fast mode. Relative to the pre-OpenGL gate (558/2 safe and 458/102 fast), the two formerly skipped parametrized 2D/3D OpenGL primitive/update/blocking-save cases now execute and pass in both modes, and the additional GIF lifecycle regression passes in both modes. Windows safe/fast real-example jobs also pass. The workflow performance benchmark completed successfully in both modes; the correction is confined to visualization process acknowledgement/GIF methods rather than computational kernels.

Headless 2D/3D OpenGL rendering, blocking save, and GIF lifecycle are therefore part of the permanent regression gate. Interactive visible-window placement/input/display lifecycle remains inherently dependent on a real display/backend and is outside the headless CI evidence.
