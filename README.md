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

- [API Guide](Documentation/API_GUIDE.md) — consolidated human-readable API reference.
- `Examples/` — complete working models.
- `Testing/` — correctness, API, integration, and regression tests.

The beginner manual, generated detailed API reference, and printable cheatsheet are under development.

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
