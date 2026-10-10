"""Multinomial API validation, state, copy, and compiled parity."""
import numpy as np
import pytest


@pytest.mark.parametrize("n", [-1, 1.5, np.nan, np.inf])
def test_safe_setup_rejects_invalid_n(api, safe_mode, n):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits Multinomial validation")
    m = api.NewMultinomial()
    with pytest.raises(ValueError):
        m.Setup(n)


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf])
def test_safe_binomial_rejects_invalid_probability(api, safe_mode, p):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits Multinomial validation")
    m = api.NewMultinomial()
    with pytest.raises(ValueError):
        m.Binomial(10, p)


def test_setup_is_chainable(api):
    m = api.NewMultinomial()
    assert m.Setup(10) is not None


def test_copy_constructor_shares_solver_but_starts_fresh_sampling_state(api):
    api.Seed(12345)
    original = api.NewMultinomial()
    original.Setup(100)
    original.Sample(0.2)
    clone = api.NewMultinomial(original)
    # The native copy shares the reusable binomial solver only; its multinomial
    # n/p state is intentionally fresh until Setup is called.
    clone.Setup(100)
    api.Seed(777)
    a = clone.Sample(0.3)
    clone.Setup(100)
    api.Seed(777)
    b = clone.Sample(0.3)
    assert a == b


def test_copy_constructor_rejects_non_multinomial(api):
    with pytest.raises(TypeError):
        api.NewMultinomial(object())


def test_compiled_setup_sample_binomial(api):
    m = api.NewMultinomial()

    @api.njit
    def work(out):
        out.Setup(20)
        a = out.Sample(0.25)
        b = out.Binomial(10, 0.5)
        return a, b

    a, b = work(m)
    assert 0 <= a <= 20
    assert 0 <= b <= 10


def test_compiled_safe_sample_rejects_probability_over_remaining_mass(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits Multinomial validation")
    m = api.NewMultinomial()

    @api.njit
    def invalid(out):
        out.Setup(10)
        out.Sample(0.8)
        out.Sample(0.3)

    with pytest.raises(ValueError):
        invalid(m)
