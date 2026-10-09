"""Multinomial exact bounds, sequential depletion, and coarse statistical checks."""
import numpy as np
import pytest


@pytest.mark.parametrize("n", [0, 1, 7, 100, 100000])
@pytest.mark.parametrize("p", [0.0, 1.0])
def test_binomial_endpoints(api, n, p):
    m = api.NewMultinomial()
    assert m.Binomial(n, p) == (n if p == 1 else 0)


def test_binomial_bounds(api):
    m = api.NewMultinomial()
    for n in (0, 1, 3, 1000):
        for p in (0.1, 0.5, 0.9):
            for _ in range(20):
                x = m.Binomial(n, p)
                assert 0 <= x <= n


def test_binomial_mean_is_plausible(api):
    m = api.NewMultinomial()
    n, p, trials = 100, 0.3, 2000
    draws = np.array([m.Binomial(n, p) for _ in range(trials)])
    # Six standard errors: robust smoke test, not a distributional certification.
    expected = n * p
    standard_error = np.sqrt(n * p * (1 - p) / trials)
    assert abs(draws.mean() - expected) < 6 * standard_error


def test_sequential_sampling_conserves_population(api):
    m = api.NewMultinomial()
    for n in (0, 1, 10, 1000):
        for _ in range(20):
            m.Setup(n)
            a = m.Sample(0.2)
            b = m.Sample(0.3)
            c = m.Sample(0.5)
            assert 0 <= a <= n
            assert 0 <= b <= n - a
            assert 0 <= c <= n - a - b
            assert a + b + c == n


@pytest.mark.parametrize("n,p", [(-1, 0.5), (1.5, 0.5), (5, -0.1), (5, 1.1)])
def test_safe_binomial_validation(api, safe_mode, n, p):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise input validation")
    m = api.NewMultinomial()
    with pytest.raises((ValueError, OverflowError)):
        m.Binomial(n, p)
