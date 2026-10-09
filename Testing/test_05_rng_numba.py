"""RNG reproducibility/range and Numba-call equivalence.

These tests deliberately avoid asserting a particular PCG32 sequence. The public
contract tested here is deterministic replay after Seed(), valid ranges, and a
shared stream across Python and PAL-compiled calls.
"""
import numpy as np
import pytest


def _python_draws(api, n=32):
    return np.asarray([api.Random() for _ in range(n)], dtype=np.float64)


def test_random_seed_replays_exact_sequence(api):
    api.Seed(123456789)
    first = _python_draws(api)
    api.Seed(123456789)
    second = _python_draws(api)
    np.testing.assert_array_equal(first, second)


def test_random_seed_changes_sequence(api):
    api.Seed(111)
    first = _python_draws(api)
    api.Seed(222)
    second = _python_draws(api)
    assert not np.array_equal(first, second)


def test_random_range(api):
    api.Seed(123)
    draws = _python_draws(api, 10000)
    assert np.all(draws >= 0.0)
    assert np.all(draws < 1.0)


@pytest.mark.parametrize("bound", [1, 2, 3, 17, 2**16 + 1, 2**32, 2**32 + 1])
def test_randint_range_and_replay(api, bound):
    api.Seed(987654321)
    first = np.asarray([api.RandInt(bound) for _ in range(200)], dtype=np.int64)
    assert np.all(first >= 0)
    assert np.all(first < bound)
    api.Seed(987654321)
    second = np.asarray([api.RandInt(bound) for _ in range(200)], dtype=np.int64)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("bound", [0, -1, 1.5, float("nan"), float("inf")])
def test_randint_rejects_invalid_bound(api, bound):
    with pytest.raises(ValueError):
        api.RandInt(bound)


@pytest.mark.parametrize("seed", [-1, 1.5, float("nan"), float("inf"), 2**64])
def test_seed_rejects_invalid_value(api, seed):
    with pytest.raises(ValueError):
        api.Seed(seed)


def test_compiled_random_matches_python_stream(api):
    @api.njit
    def compiled_draws(n):
        out = np.empty(n)
        for i in range(n):
            out[i] = api.Random()
        return out

    api.Seed(314159)
    expected = _python_draws(api, 24)
    api.Seed(314159)
    observed = compiled_draws(24)
    np.testing.assert_array_equal(observed, expected)


def test_compiled_randint_matches_python_stream(api):
    @api.njit
    def compiled_draws(n, bound):
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = api.RandInt(bound)
        return out

    api.Seed(271828)
    expected = np.asarray([api.RandInt(97) for _ in range(50)], dtype=np.int64)
    api.Seed(271828)
    observed = compiled_draws(50, 97)
    np.testing.assert_array_equal(observed, expected)


def test_ilist_random_uses_seeded_pal_stream(api):
    q = api.NewIList()
    for value in (3, 7, 11, 19, 23):
        q.Append(value)
    api.Seed(4242)
    first = [int(q.Random()) for _ in range(40)]
    api.Seed(4242)
    second = [int(q.Random()) for _ in range(40)]
    assert first == second
