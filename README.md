# Python Automata Library (PAL)

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

## Documentation

- [PAL Guide](Documentation/PAL_GUIDE.md) ([PDF](Documentation/PAL_GUIDE.pdf)) — first model, worked modeling workflows, and detailed API behavior.
- [PAL Architecture and Design](Documentation/PAL_ARCHITECTURE_AND_DESIGN.md) ([PDF](Documentation/PAL_ARCHITECTURE_AND_DESIGN.pdf)) — implementation structure, design rationale, subsystem choices, testing, and performance.
- [API Reference](Documentation/API_REFERENCE.md) ([PDF](Documentation/API_REFERENCE.pdf)) — source-derived public signatures.
- [Cheatsheet](Documentation/CHEATSHEET.md) ([PDF](Documentation/CHEATSHEET.pdf)) — compact one-page lookup for common operations.
- `Examples/` — complete working models.
- [Testing](Testing/README.md) — correctness, API, integration, and regression tests, with local and CI run instructions.

## Installation

Install PAL from PyPI:

```text
python -m pip install PythonAutomataLibrary
```

Release wheels are intended to bundle the native core for Windows, Linux, and macOS, without manual compilation. PyPI publication is pending; see [distribution status](Documentation/DISTRIBUTION.md).

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

> Bravo, R. R. (2026). *Python Automata Library (PAL)* [Computer software]. https://github.com/rafaelrbravo/PythonAutomataLibrary

GitHub's **Cite this repository** button provides citation formats from [CITATION.cff](CITATION.cff).

PAL is based on the Hybrid Automata Library (HAL). Please also acknowledge or cite HAL when its underlying ideas or algorithms are relevant to your work.
