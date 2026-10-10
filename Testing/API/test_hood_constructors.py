"""Neighborhood constructor contract tests beyond exact geometry checks."""
import numpy as np
import pytest


@pytest.mark.parametrize("fn", ["MooreHood", "VonNeumannHood"])
@pytest.mark.parametrize("dim", [1.0, np.int64(2), True, "2"])
def test_discrete_hood_dimension_contract(api, fn, dim):
    call = getattr(api, fn)
    if isinstance(dim, np.integer) or dim == 1.0 and not isinstance(dim, (bool, np.bool_)):
        # Current API accepts numerically equal integral dimensions through
        # membership comparison; preserve that behavior explicitly.
        hood = call(dim)
        assert hood
    else:
        with pytest.raises(ValueError):
            call(dim)


@pytest.mark.parametrize("exclude", [False, True])
def test_circle_radius_zero_center_contract(api, exclude):
    hood = api.CircleHood(2, 0, excludeCenter=exclude)
    assert hood == (() if exclude else ((0, 0),))


@pytest.mark.parametrize("radius", [0.5, 1.25, np.float32(2.0)])
def test_circle_radius_is_euclidean_not_integer_truncated(api, radius):
    hood = api.CircleHood(2, radius)
    for x, y in hood:
        assert x*x + y*y <= float(radius)*float(radius)


@pytest.mark.parametrize("radius", [True, "1", None, -0.1, np.nan, np.inf])
def test_circle_rejects_invalid_radius_types_and_values(api, radius):
    with pytest.raises(ValueError):
        api.CircleHood(2, radius)
