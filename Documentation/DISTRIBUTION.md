# PAL distribution and release checklist

PAL targets **Windows, Linux, and macOS**. Public installation through PyPI and conda-forge is **not confirmed**. Do not advertise the commands below as working until packages are published and installation is tested.

## Build locally

From the repository root, with a C compiler on PATH (MSVC Developer Command Prompt on Windows):

```bash
python -m pip install --upgrade build
python -m build
```

The resulting `dist/` directory contains a platform-specific wheel and a source archive. The wheel must contain `pal_native.dll` (Windows), `libpal_native.so` (Linux), or `libpal_native.dylib` (macOS). Source archives must contain `pal_native.c` so they can compile during installation.

To check a wheel without importing from the checkout, install it in a new environment, change directory outside the repository, and run `packaging/smoke_install.py` using that environment's Python interpreter.

## Continuous integration

- `.github/workflows/package-smoke.yml` builds and tests wheels and source distributions on Windows, Linux, and macOS.
- `.github/workflows/conda-smoke.yml` builds and tests the local conda recipe on Windows, Linux, and macOS; this is not a conda-forge publication.
- `.github/workflows/release-build.yml` builds versioned wheel artifacts for CPython 3.10–3.13 using cibuildwheel and a source archive; it runs on relevant packaging changes to `main`, manual dispatch, and `v*` tags. On a `v*` tag, successful builds are collected into a **draft** GitHub Release with SHA256 checksums; publishing the draft requires review.
- Before publishing, inspect every job result, install wheels on each supported platform, and confirm the bundled native library loads and PAL tests pass.

## Release procedure (pending validation)

1. Confirm the intended version matches `pyproject.toml` and the Git tag, for example `v0.1.0`.
2. Ensure the platform build matrix passes and inspect wheel tags and artifacts. Do not publish a wheel tagged `none-any`.
3. Download build artifacts and verify filenames, architectures, and native library inclusion.
4. Inspect the draft GitHub Release created after successful tagged builds, verify wheel assets and SHA256 checksums, then publish the draft.
5. Configure a PyPI Trusted Publisher for GitHub repository `rafaelrbravo/PythonAutomataLibrary`, workflow `pypi-publish.yml`, environment `pypi`. After publishing the GitHub Release, manually dispatch `.github/workflows/pypi-publish.yml` with the published version tag. The workflow verifies SHA256 checksums and publishes the **same tested artifacts** using OpenID Connect. Do not embed a PyPI token in source code.
6. Replace the local-path conda recipe in `packaging/conda/meta.yaml` with the published source archive URL and SHA256. Submit to conda-forge, then validate installs on all three OSes.
7. After publication and verification, document supported install commands:

```bash
pip install PythonAutomataLibrary
pip install 'PythonAutomataLibrary[visualization]'  # optional GUI dependencies
conda install conda-forge::pythonautomatalibrary
```

## Release blockers

**Verified:** Package-smoke passed on Windows, Linux, and macOS for commit `a2852eac` ([run 38346812005](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38346812005)). Conda-smoke passed on all three platforms for commit `bac272e4` ([run 38346999220](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38346999220)). These checks validate local builds and tests, not public package availability.

**Release matrix:** The full CPython 3.10–3.13 matrix passed on Linux, Windows AMD64, macOS ARM64, and macOS Intel in [run 38347630189](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38347630189), and again in [run 38347711431](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38347711431). Later validation changes, including per-platform wheel completeness checks, still require a successful current-head run.

**Additional completed wheel matrix:** All four wheel jobs (Linux, Windows, macOS Intel, macOS ARM64) passed in [run 38348218337](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38348218337). The same run's `sdist` job failed, so the run as a whole is **not** release-ready; it predates the source-archive fixes.

**Known source-archive failure:** [run 38348218337](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/runs/38348218337) failed its `sdist` job because the archive contained `pal_native.dll` and `libpal_native.so`. The source archive must ship `pal_native.c` and build helpers, **not** precompiled native libraries. `MANIFEST.in` now uses recursive exclusions and `setup.py` filters the `sdist` file list; these changes have not yet passed a full CI rebuild. Do not use the failed run's source archive for release.

**Latest packaging hardening (2026-10-13):** `MANIFEST.in` and the custom `sdist` command exclude compiled libraries by suffix. The release-build, draft GitHub Release, PyPI publishing, and cross-platform package-smoke archive checks now reject `.dll`, `.so`, `.dylib`, and `.pyd` files and require essential build sources. These are committed safeguards, **not proof of a successful current-head CI run**. The PyPI workflow is manually dispatched only after a published GitHub Release, and no publication is authorized yet.

**Next validation gate:** Open [release-build Actions](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/workflows/release-build.yml) and inspect the newest run at the latest `main` commit. If there is no suitable run, use **Run workflow** on `main` (no release tag needed). Require `version-check`, `sdist` (including installation, `pip check`, and the native smoke test of the rebuilt wheel), all four `wheels` platform jobs, and `verify-release-assets` to succeed. The aggregate verification now runs on `main` without creating a release, checks the complete 16-wheel/source-archive set, and generates SHA256SUMS.txt; the draft-release creation step remains tag-only. Then inspect/download the source archive and 16 wheels. Also verify the newest [package-smoke run](https://github.com/rafaelrbravo/PythonAutomataLibrary/actions/workflows/package-smoke.yml), especially the rebuilt source-archive wheels on all three operating systems. Do not tag or publish until both checks pass.

**CI queue behavior:** The release-build workflow now cancels superseded builds on `main` using per-ref concurrency, but preserves tagged release builds. For validation, use the run associated with the latest commit rather than a successful job from an older run. Older passing matrices do not establish current-head release readiness.

**Still blocked:** Confirm the latest stricter release matrix and source-archive validation, review and publish GitHub Release assets, configure PyPI Trusted Publishing, and submit a source-archive-based conda-forge recipe. The local conda recipe is a starting point, not a published feedstock.
