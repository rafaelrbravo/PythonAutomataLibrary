"""Independent exact-set checks for 1D/2D/3D neighborhood geometry."""
import itertools

import pytest


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("exclude", [False, True])
def test_moore_offsets(api, dim, exclude):
    expected = set(itertools.product((-1, 0, 1), repeat=dim))
    if exclude:
        expected.remove((0,) * dim)
    actual = api.MooreHood(dim, exclude)
    assert len(actual) == len(expected)
    assert set(actual) == expected


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("exclude", [False, True])
def test_von_neumann_offsets(api, dim, exclude):
    expected = {(0,) * dim} if not exclude else set()
    for axis in range(dim):
        for sign in (-1, 1):
            v = [0] * dim
            v[axis] = sign
            expected.add(tuple(v))
    actual = api.VonNeumannHood(dim, exclude)
    assert len(actual) == len(expected)
    assert set(actual) == expected


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("radius", [0, 1, 1.5, 2])
@pytest.mark.parametrize("exclude", [False, True])
def test_circle_offsets(api, dim, radius, exclude):
    extent = int(radius)
    expected = {
        point for point in itertools.product(range(-extent, extent + 1), repeat=dim)
        if sum(x*x for x in point) <= radius*radius
        and (not exclude or any(point))
    }
    actual = api.CircleHood(dim, radius, exclude)
    assert len(actual) == len(expected)
    assert set(actual) == expected


@pytest.mark.parametrize("constructor", ["MooreHood", "VonNeumannHood", "CircleHood"])
@pytest.mark.parametrize("dim", [0, 4])
def test_invalid_dimensions(api, constructor, dim):
    fn = getattr(api, constructor)
    with pytest.raises(ValueError):
        fn(dim, 1) if constructor == "CircleHood" else fn(dim)


@pytest.mark.parametrize("radius", [-1, float("nan"), float("inf")])
def test_invalid_circle_radius(api, radius):
    with pytest.raises(ValueError):
        api.CircleHood(2, radius)
