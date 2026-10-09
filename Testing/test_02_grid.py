"""Grid invariants: row-major indexing, typed storage, copy-on-slice, wrapping."""
import itertools

import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 2)])
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_row_major_coordinates_and_storage(api, shape, dtype):
    g = api.NewGrid(shape, dtype)
    assert g.nDims == len(shape)
    assert len(g) == int(np.prod(shape))
    for coordinates in itertools.product(*(range(n) for n in shape)):
        linear = int(np.ravel_multi_index(coordinates, shape, order="C"))
        assert g.ToI(*coordinates) == linear
        assert g.ItoX(linear) == coordinates[0]
        if len(shape) > 1:
            assert g.ItoY(linear) == coordinates[1]
        if len(shape) > 2:
            assert g.ItoZ(linear) == coordinates[2]
        value = linear + 1
        g[coordinates] = value
        assert g[linear] == value
        assert g[coordinates] == value


@pytest.mark.parametrize("shape", [(5,), (4, 3), (3, 4, 2)])
def test_slices_are_copies_and_match_numpy(api, shape):
    g = api.NewGrid(shape, np.int32)
    ref = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    for i, value in enumerate(ref.flat):
        g[i] = int(value)
    key = tuple(slice(1, None) for _ in shape)
    observed = g[key]
    np.testing.assert_array_equal(observed, ref[key])
    observed[...] = -99
    np.testing.assert_array_equal(g[key], ref[key])


@pytest.mark.parametrize("dims,wrapped", [
    ((5,), (False,)),
    ((-5,), (True,)),
    ((4, -3), (False, True)),
    ((-4, 3, -2), (True, False, True)),
])
def test_wrap_coordinate_contract(api, dims, wrapped):
    g = api.NewGrid(dims, np.float64)
    for axis, (dim, wraps) in enumerate(zip(dims, wrapped)):
        fn = (g.InWrapX, g.InWrapY, g.InWrapZ)[axis]
        size = abs(dim)
        for coordinate in (-2*size-1, -1, 0, size-1, size, 2*size+1):
            expected = coordinate % size if wraps else (coordinate if 0 <= coordinate < size else -1)
            assert fn(coordinate) == expected


@pytest.mark.parametrize("dtype,value", [(np.int8, 127), (np.uint8, 255), (np.float32, 1.25), (np.bool_, 1)])
def test_representable_values(api, dtype, value):
    g = api.NewGrid((3,), dtype)
    g[1] = value
    assert g[1] == value


@pytest.mark.parametrize("dims", [(), (0,), (1, 2, 3, 4), (1.5,), (float("nan"),)])
def test_invalid_dimensions(api, dims):
    with pytest.raises(ValueError):
        api.NewGrid(dims, np.int32)


def test_safe_rejects_invalid_indices_and_values(api, safe_mode):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise input validation")
    g = api.NewGrid((4, 3), np.uint8)
    for key in [-1, 12, (4, 0), (0, -1), (0, 0, 0)]:
        with pytest.raises((IndexError, ValueError)):
            _ = g[key]
    for value in (-1, 256, 1.5, float("nan")):
        with pytest.raises(ValueError):
            g[0] = value
