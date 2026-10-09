# PAL native core

Readable C + thin Numba jitclass architecture for PAL.

Design rules:
- C owns mutable simulation state and workhorse algorithms.
- Safe jitclasses contain only the native pointer plus immutable metadata needed for validation/API dispatch.
- Fast jitclasses omit safety-only metadata where possible and call unchecked C entry points directly.
- No runtime safe/fast branch exists in JIT-visible methods.
- User models remain ordinary `@njit` code with the object-oriented PAL API.

## Build
Windows, from an x64 Visual Studio Developer Command Prompt:

    python build_native.py

Linux/macOS:

    python build_native.py

Then:

    python test_native.py
    python BirthDeathNative.py

## Current coverage
This package contains the completed native-core architecture and tested central paths for QueryList, AgentGrid, PopGrid, and PDEgrid. It is not yet a claim of full API parity with the three reference PAL files. In particular, the reference implementations still contain method families that require faithful native ports (not stubs): AgentGrid radius/box helpers and some neighborhood generators, PopGrid's complete Binomial helper/integer-width variants, and PDEgrid's ADI/advection/radial/mask/field/interface diffusion families.

The package intentionally leaves those as explicit remaining work rather than hiding Python fallbacks or generated C behind the native API.
