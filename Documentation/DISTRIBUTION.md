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
- `.github/workflows/release-build.yml` builds versioned wheel artifacts for CPython 3.10–3.13 using cibuildwheel and a source archive. On a `v*` tag, successful builds are collected into a **draft** GitHub Release with SHA256 checksums; publishing the draft requires review.
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

The CI workflows have not yet been confirmed passing, release assets have not been verified or published, PyPI Trusted Publishing is not confirmed configured, and the conda-forge feedstock has not been submitted. The local conda recipe is a starting point, not a published feedstock.
