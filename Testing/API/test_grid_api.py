"""Grid API dtype, slicing, compiled, and diagnostic contracts."""
import numpy as np
import pytest
import PythonAutomataLibrary as pal


@pytest.mark.parametrize("dtype,value", [
    (np.int8, -7), (np.int16, -300), (np.int32, -70000), (np.int64, -(2**40)),
    (np.uint8, 250), (np.uint16, 60000), (np.uint32, 2**31), (np.uint64, 2**63),
    (np.float32, 1.25), (np.float64, -3.5), (np.bool_, True),
])
def test_scalar_roundtrip_all_supported_dtypes(api, dtype, value):
    g = api.NewGrid((3,), dtype)
    g[1] = value
    assert g[1] == value


@pytest.mark.parametrize("shape,key", [
    ((6,), slice(1, 5, 2)),
    ((4, 5), (slice(1, 4), slice(None, None, 2))),
    ((3, 4, 5), (slice(None), 2, slice(1, 5, 2))),
])
def test_multidimensional_slice_get_set_matches_numpy(api, shape, key):
    ref = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    g = api.NewGrid(shape, np.int32)
    g[tuple(slice(None) for _ in shape)] = ref
    np.testing.assert_array_equal(g[key], ref[key])
    replacement = np.full(ref[key].shape, -9, dtype=np.int32)
    g[key] = replacement
    ref[key] = replacement
    np.testing.assert_array_equal(g[tuple(slice(None) for _ in shape)], ref)


def test_compiled_unannotated_slice_get_and_set(api):
    g = api.NewGrid((4, 5), np.int32)
    g[:, :] = np.arange(20, dtype=np.int32).reshape(4, 5)

    @api.njit
    def work(grid):
        before = grid[1:4, 1:5:2]
        grid[0:2, 2:5] = 7
        return before

    got = work(g)
    np.testing.assert_array_equal(got, np.array([[6, 8], [11, 13], [16, 18]], dtype=np.int32))
    expected = np.arange(20, dtype=np.int32).reshape(4, 5)
    expected[0:2, 2:5] = 7
    np.testing.assert_array_equal(g[:, :], expected)


def test_compiled_keyword_toi_and_wrap(api):
    g = api.NewGrid((-4, -5, -6), np.int32)

    @api.njit
    def work(grid):
        return grid.ToI(x=2, y=3, z=4), grid.InWrapX(-1), grid.InWrapY(5), grid.InWrapZ(7)

    assert work(g) == (g.ToI(2, 3, 4), 3, 0, 1)


def test_compiled_safe_invalid_linear_getitem_rejected(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits Grid bounds validation")
    g = api.NewGrid((4, 3), np.int32)

    @api.njit
    def read(grid, i):
        return grid[i]

    for i in (-1, 12):
        with pytest.raises((IndexError, ValueError)):
            read(g, i)


def test_compiled_safe_invalid_coordinate_getitem_rejected(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits Grid bounds validation")
    g = api.NewGrid((4, 3), np.int32)

    @api.njit
    def read(grid, x, y):
        return grid[x, y]

    for x, y in ((4, 0), (0, -1)):
        with pytest.raises((IndexError, ValueError)):
            read(g, x, y)


def test_annotated_safe_scalar_set_error_reports_source_line(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode omits validation diagnostics")
    g = api.NewGrid((4,), np.int32)

    @api.njit
    def invalid(grid: pal.Grid):
        grid[4] = 1

    with pytest.raises(IndexError, match="source line"):
        invalid(g)


def test_compiled_grid_inwrap_keywords_regression(api):
    g = api.NewGrid((-4, -5, -6), np.int32)

    @api.njit
    def work(grid):
        return grid.InWrapX(x=-1), grid.InWrapY(y=5), grid.InWrapZ(z=7)

    assert work(g) == (3, 0, 1)
