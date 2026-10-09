"""Stronger deterministic/statistical checks for PAL's public RNG surface.

These remain smoke tests rather than a certification suite. Fixed seeds make
failures reproducible; tolerances are intentionally loose enough to avoid
flaky CI while detecting severe bias or broken high-bound paths.
"""
import numpy as np
import pytest


def test_random_uniform_moments_and_quartiles(api):
    api.Seed(0xC0FFEE)
    x = np.asarray([api.Random() for _ in range(50000)], dtype=np.float64)
    assert abs(float(x.mean()) - 0.5) < 0.008
    assert abs(float(x.var()) - 1.0 / 12.0) < 0.004
    counts, _ = np.histogram(x, bins=[0.0, 0.25, 0.5, 0.75, 1.0])
    expected = len(x) / 4
    # ~5.5 sigma per quartile under a multinomial approximation.
    assert np.max(np.abs(counts - expected)) < 600


@pytest.mark.parametrize("bound", [3, 10, 257, 65537])
def test_randint_small_and_medium_bounds_are_roughly_uniform(api, bound):
    api.Seed(19000 + bound)
    n = 40000
    # For large bounds, bucket modulo a fixed divisor only when it divides the
    # bound poorly enough that exact bucket probabilities are computed below.
    buckets = min(bound, 16)
    draws = np.asarray([api.RandInt(bound) for _ in range(n)], dtype=np.int64)
    counts = np.bincount(draws * buckets // bound, minlength=buckets)
    # Exact bucket widths in integer outcome space.
    widths = np.array([
        ((j + 1) * bound + buckets - 1) // buckets - (j * bound + buckets - 1) // buckets
        for j in range(buckets)
    ], dtype=np.float64)
    probs = widths / bound
    expected = n * probs
    sigma = np.sqrt(n * probs * (1 - probs))
    assert np.all(np.abs(counts - expected) <= 6.0 * sigma + 3.0)


@pytest.mark.parametrize("bound", [2**32, 2**32 + 1, 2**40 + 123])
def test_randint_large_bound_exercises_upper_32_bits(api, bound):
    api.Seed(20261009)
    draws = np.asarray([api.RandInt(bound) for _ in range(5000)], dtype=np.int64)
    assert np.all((0 <= draws) & (draws < bound))
    if bound > 2**32:
        # A broken 32-bit-only implementation can never reach this region.
        assert np.any(draws >= 2**32)


def test_random_lag1_correlation_smoke(api):
    api.Seed(1234567)
    x = np.asarray([api.Random() for _ in range(30000)], dtype=np.float64)
    corr = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    assert abs(corr) < 0.04
