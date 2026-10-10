"""Module RNG boundary and compiled argument contracts."""
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 2**32, 2**64 - 1, np.uint64(17)])
def test_seed_accepts_uint64_domain(api, seed):
    api.Seed(seed)
    value = api.Random()
    assert 0.0 <= value < 1.0


@pytest.mark.parametrize("bound", [1, 2**31, 2**63 - 1])
def test_randint_accepts_positive_int64_domain(api, bound):
    api.Seed(5)
    value = api.RandInt(bound)
    assert 0 <= value < bound


@pytest.mark.parametrize("bound", [0, -1, 1.5, np.nan, np.inf, 2**63])
def test_compiled_randint_rejects_invalid_bound(api, bound):
    @api.njit
    def draw(x):
        return api.RandInt(x)

    with pytest.raises(ValueError):
        draw(bound)


def test_random_and_randint_share_single_seeded_stream_across_call_boundaries(api):
    @api.njit
    def middle():
        return api.Random(), api.RandInt(101)

    api.Seed(2026)
    first = api.Random()
    mid = middle()
    last = api.RandInt(101)

    api.Seed(2026)
    expected = (api.Random(), api.Random(), api.RandInt(101), api.RandInt(101))
    assert (first, mid[0], mid[1], last) == expected
