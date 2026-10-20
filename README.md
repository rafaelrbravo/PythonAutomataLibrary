# Python Automata Library (PAL)

[![DOI](https://zenodo.org/badge/1385841661.svg)](https://doi.org/10.5281/zenodo.23439298)

PAL is a Python modeling library for high-performance agent-based, lattice population, and PDE models. Model code uses a compact object-oriented API and can be compiled with `@pal.njit`; PAL keeps performance-sensitive state and algorithms in a precompiled native core.

```python
import PythonAutomataLibrary as pal
```

PAL provides:

- `Grid`: typed 1D/2D/3D lattice data.
- `AgentGrid`: lattice or continuous-position agents with identity and properties.
- `PopGrid`: integer lattice populations without individual agent identity.
- `PDEgrid`: diffusion/advection fields.
- `IList`: reusable integer query results.
- `Multinomial`: binomial/multinomial sampling.
- `PixWindow` and `OpenGLWindow`: 2D/3D visualization.
- `@pal.njit`: Numba compilation with PAL-specific source transformation and diagnostics.

PAL runs in safe mode by default. Call `pal.FastMode()` before constructing PAL objects to select the unchecked high-performance implementations after a model has been validated.

## Quick start

Install PAL:

```bash
python -m pip install PythonAutomataLibrary
```

Copy the bundled examples and PDF/Markdown documentation to an editable `PAL-Examples/` folder in your current directory:

```bash
python -m PythonAutomataLibrary.ExampleInstaller
```

Run an example script from `PAL-Examples/`. For a guided first model, see the [PAL Guide](Documentation/PAL_GUIDE.md).

## Documentation

- [PAL Guide](Documentation/PAL_GUIDE.md) ([PDF](Documentation/PAL_GUIDE.pdf)) — first model, worked modeling workflows, and detailed API behavior.
- [PAL Architecture and Design](Documentation/PAL_ARCHITECTURE_AND_DESIGN.md) ([PDF](Documentation/PAL_ARCHITECTURE_AND_DESIGN.pdf)) — implementation structure, design rationale, subsystem choices, testing, and performance.
- [API Reference](Documentation/API_REFERENCE.md) ([PDF](Documentation/API_REFERENCE.pdf)) — source-derived public signatures.
- [Cheatsheet](Documentation/CHEATSHEET.md) ([PDF](Documentation/CHEATSHEET.pdf)) — compact one-page lookup for common operations.
- `Examples/` — complete working models.
- [Testing](Testing/README.md) — correctness, API, integration, and regression tests, with local and CI run instructions.

## Installation details

PyPI wheels include the native core for Windows, Linux, and macOS, so manual compilation is unnecessary. The standard installation includes visualization dependencies. See [PAL on PyPI](https://pypi.org/project/PythonAutomataLibrary/) and [distribution details](Documentation/DISTRIBUTION.md).

Advanced users who only need a minimal headless installation can skip declared dependencies and install the core requirements explicitly:

```bash
python -m pip install --no-deps PythonAutomataLibrary
python -m pip install numpy numba
```

Note that `--no-deps` skips *all* declared dependencies, not only visualization. As a result, `pip check` may report missing dependencies.

To install examples in a different parent directory, pass its path, such as `python -m PythonAutomataLibrary.ExampleInstaller ~/Documents` or `python -m PythonAutomataLibrary.ExampleInstaller "C:\\Users\\YourName\\Documents"` on Windows. The installer creates `PAL-Examples/` with runnable scripts and a `Documentation/` directory of PDF and Markdown manuals, without downloading from GitHub; it refuses to overwrite an existing `PAL-Examples/` folder. Graphical examples require a desktop display.

## Build from source

PAL currently loads its native library from the package directory. Build it before importing PAL.

Windows, from an x64 Visual Studio Developer Command Prompt:

```text
python build_native.py
```

Linux/macOS:

```text
python build_native.py
```

Then run the tests:

```text
python -m pytest -q Testing
```

## Architecture

`NativeCore.py` is the Python/Numba API and compilation layer. `pal_native.c` owns stable mutable simulation state and the workhorse native algorithms. Safe and fast PAL objects are selected at construction time, so compiled hot paths do not branch on safety mode.

The public API has dedicated regression coverage in `Testing/API/`; the full `Testing/` tree is the integration gate.

## Citation

If you use PAL in your research, please cite:

> Bravo, R. R. (2026). *Python Automata Library (PAL)* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23439298

For reproducibility, cite the specific release you used: [PAL v1.0.0](https://doi.org/10.5281/zenodo.23439299).

GitHub's **Cite this repository** button provides citation formats from [CITATION.cff](CITATION.cff).

PAL is based on the Hybrid Automata Library (HAL). Please also acknowledge or cite HAL when its underlying ideas or algorithms are relevant to your work.
