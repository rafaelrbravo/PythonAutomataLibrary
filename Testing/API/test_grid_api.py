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


@pytest.mark.parametrize("dtype,bad_values", [
    (np.bool_, (2, -1)),
    (np.int8, (1.5, 128, -129, np.inf)),
    (np.uint8, (-1, 256, np.nan)),
    (np.float32, (np.inf, -np.inf, np.nan, 1e100)),
])
def test_python_safe_grid_rejects_unrepresentable_assignments(api, safe_mode, dtype, bad_values):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits dtype validation")
    g = api.NewGrid((3,), dtype)
    g[1] = 1
    for value in bad_values:
        with pytest.raises(ValueError, match="not representable"):
            g[1] = value
        assert g[1] == 1


@pytest.mark.parametrize("factory,dtype", [("NewPopGrid", np.int64), ("NewPDEgrid", np.float32)])
def test_population_and_pde_python_slices_return_detached_arrays(api, factory, dtype):
    grid = getattr(api, factory)((4, 5))
    grid[2, 3] = 7
    region = grid[:, :]
    assert region.shape == (4, 5)
    assert region.dtype == dtype
    assert region[2, 3] == 7
    region[2, 3] = 99
    assert grid[2, 3] == 7
    linear = grid[0:len(grid)]
    assert linear.shape == (20,)
    assert linear[grid.ToI(2, 3)] == 7


@pytest.mark.parametrize("factory,dtype", [("NewPopGrid", np.int64), ("NewPDEgrid", np.float32)])
def test_population_and_pde_python_array_slice_assignment(api, factory, dtype):
    grid = getattr(api, factory)((4, 5))
    values = np.arange(20, dtype=dtype).reshape(4, 5)
    grid[:, :] = values
    np.testing.assert_array_equal(grid[:, :], values)
    replacement = np.full(20, 3, dtype=dtype)
    grid[0:len(grid)] = replacement
    np.testing.assert_array_equal(grid[:, :], replacement.reshape(4, 5))


def test_python_box_half_open_bounds_and_wrapping(api):
    grid = api.NewGrid((-3, 4), np.int32)
    assert list(grid.Box(2, 5, -1, 2)) == [
        (2, 0), (2, 1), (0, 0), (0, 1), (1, 0), (1, 1)
    ]
    # A wrapped box larger than its axis can visit the same site more than once.
    assert list(grid.Box(0, 5, 1, 2)) == [(0, 1), (1, 1), (2, 1), (0, 1), (1, 1)]
    assert list(api.NewGrid((3,), np.int32).Box(-2, 5)) == [0, 1, 2]


def test_python_box_rejects_invalid_bounds(api):
    grid = api.NewGrid((3, 4), np.int32)
    with pytest.raises(ValueError, match="dimensionality"):
        list(grid.Box(0, 2))
    with pytest.raises(TypeError, match="Box expects"):
        list(grid.Box(0, 2, 0))
    for bad in (1.5, True, 2**40):
        with pytest.raises(ValueError, match="int32"):
            list(grid.Box(0, bad, 0, 2))


def test_python_hood_maps_offsets_to_coordinates(api):
    grid = api.NewGrid((-3, 4), np.int32)
    hood = ((-1, 0), (0, 0), (1, 0), (0, -1), (0, 1))
    assert list(grid.Hood(hood, 0, 0)) == [(2, 0), (0, 0), (1, 0), (0, 1)]
    # Offset order and duplicates are retained; Hood does not deduplicate wrapped sites.
    assert list(grid.Hood(((0, 0), (3, 0)), 1, 2)) == [(1, 2)] * 2
    with pytest.raises(TypeError, match="tuple"):
        list(grid.Hood([(-1, 0)], 1, 2))
    with pytest.raises(ValueError, match="dimensionality"):
        list(grid.Hood(((1,),), 1, 2))


@pytest.mark.parametrize("shape", [(7,), (5, 6), (4, 5, 6)])
def test_python_njit_grid_scalar_mutation_parity(api, shape):
    """Compare coordinate assignment, linear indexing, and full-array state."""
    py = api.NewGrid(shape, np.int32)
    jit = api.NewGrid(shape, np.int32)

    def python_work(g):
        site = tuple(1 for _ in shape)
        key = site if len(shape) > 1 else site[0]
        g[key] = 11
        index = g.ToI(*site)
        before = g[index]
        g[index] = -4
        return index, before, g[key]

    if len(shape) == 1:
        @api.njit
        def compiled_work(g):
            g[1] = 11
            index = g.ToI(1)
            before = g[index]
            g[index] = -4
            return index, before, g[1]
    elif len(shape) == 2:
        @api.njit
        def compiled_work(g):
            g[1, 1] = 11
            index = g.ToI(1, 1)
            before = g[index]
            g[index] = -4
            return index, before, g[1, 1]
    else:
        @api.njit
        def compiled_work(g):
            g[1, 1, 1] = 11
            index = g.ToI(1, 1, 1)
            before = g[index]
            g[index] = -4
            return index, before, g[1, 1, 1]

    assert compiled_work(jit) == python_work(py)
    np.testing.assert_array_equal(jit[:], py[:])


@pytest.mark.parametrize("bad_index", [-1, 12])
def test_python_njit_grid_invalid_linear_read_atomic_parity(api, safe_mode, bad_index):
    """Safe mode should reject invalid reads without changing grid contents."""
    if not safe_mode:
        pytest.skip("Fast mode deliberately omits bounds checks")
    py = api.NewGrid((4, 3), np.int32)
    jit = api.NewGrid((4, 3), np.int32)
    py[:, :] = np.arange(12, dtype=np.int32).reshape(4, 3)
    jit[:, :] = np.arange(12, dtype=np.int32).reshape(4, 3)

    @api.njit
    def compiled_read(g, index):
        return g[index]

    with pytest.raises((IndexError, ValueError)) as python_error:
        _ = py[bad_index]
    with pytest.raises((IndexError, ValueError)) as compiled_error:
        compiled_read(jit, bad_index)
    assert type(python_error.value) is type(compiled_error.value)
    np.testing.assert_array_equal(jit[:, :], py[:, :])


@pytest.mark.parametrize("coordinates", [(4, 0), (0, -1), (0, 3)])
def test_python_njit_grid_invalid_coordinate_read_atomic_parity(api, safe_mode, coordinates):
    """Compare safe-mode coordinate-read exception types and state preservation."""
    if not safe_mode:
        pytest.skip("Fast mode deliberately omits bounds checks")
    py = api.NewGrid((4, 3), np.int32)
    jit = api.NewGrid((4, 3), np.int32)
    initial = np.arange(12, dtype=np.int32).reshape(4, 3)
    py[:, :] = initial
    jit[:, :] = initial

    @api.njit
    def compiled_read(g, x, y):
        return g[x, y]

    with pytest.raises((IndexError, ValueError)) as python_error:
        _ = py[coordinates]
    with pytest.raises((IndexError, ValueError)) as compiled_error:
        compiled_read(jit, *coordinates)
    assert type(python_error.value) is type(compiled_error.value)
    np.testing.assert_array_equal(py[:, :], initial)
    np.testing.assert_array_equal(jit[:, :], initial)
