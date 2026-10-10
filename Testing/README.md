# PAL testing

`Testing/` is PAL's executable regression and correctness suite. It covers public behavior across neighborhood geometry, Grid/IList/Multinomial/RNG operations, AgentGrid and PopGrid state, PDE transport and boundaries, Python/`@pal.njit` parity, visualization, integrated models, stress cases, and historical regressions.

Tests should establish public behavior with independent invariants or small reference calculations where practical. Prefer conservation laws, coordinate identities, exact hand-checkable cases, and Python/compiled agreement over reproducing PAL's implementation inside the test.

## Run the suite

Build or install a compatible PAL native library, then run safe and fast modes in separate Python processes from the repository root:

```bash
PAL_TEST_MODE=safe python -m pytest -q Testing
PAL_TEST_MODE=fast python -m pytest -q Testing
```

PAL locks its process-wide mode at first construction. Fast mode intentionally omits selected validation checks, so some invalid-input tests are safe-only.

For a focused change, run the affected test file first:

```bash
PAL_TEST_MODE=safe python -m pytest -q Testing/API/test_pde_advection.py
```

Then run the full suite in both modes.

## Continuous integration

`.github/workflows/pal-api-audit.yml` is the authoritative CI recipe. On Linux it:

1. installs the Python test and headless-rendering dependencies;
2. verifies `Documentation/API_REFERENCE.md` matches the generator;
3. builds `libpal_native.so`;
4. runs the full `Testing/` suite in safe and fast modes;
5. runs the API performance benchmark after successful correctness tests.

Separate Windows jobs build `pal_native.dll` with MSVC and run representative real-example regressions in safe and fast modes. Headless OpenGL execution requires a compatible graphics backend.

## Performance

Correctness tests and performance measurements are separate. The CI API benchmark runs `Testing/benchmark_keyword_binding.py` repeatedly after the test suite passes. Additional benchmarks under `Testing/` can be used for targeted profiling. Compare repeated warm measurements, and distinguish compilation/startup cost from steady-state runtime.

## Adding tests

Keep regressions minimal and public-facing. A useful test should make the intended contract obvious and fail for the defect it protects against. Exercise boundaries and wrapping where relevant, and test both Python and compiled paths when both are supported. Safe-mode failure tests should also verify transactionality when an operation promises to reject invalid input before mutating pending or current state.
