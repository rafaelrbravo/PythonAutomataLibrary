"""Grid assignment parity tests; run safe and fast in separate processes.

The Persian.py compiled whole-grid slice defect is kept as a strict expected
failure until corrected; the scalar-index loop is a supported workaround.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float32])
def test_python_whole_grid_slice_scalar_assignment(api, shape, dtype):
    grid = api.NewGrid(shape, dtype)
    grid[:] = 3
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(np.prod(shape), 3, dtype=dtype))


@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 4)])
def test_compiled_scalar_integer_assignment_control(api, shape):
    grid = api.NewGrid(shape, np.int8)

    @api.njit
    def fill(g):
        for i in range(len(g)):
            g[i] = 2

    fill(grid)
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(np.prod(shape), 2, dtype=np.int8))


def test_compiled_whole_grid_slice_scalar_assignment_unannotated(api):
    grid = api.NewGrid((3, 5), np.int8)

    @api.njit
    def fill(g):
        g[:] = 2

    fill(grid)
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(15, 2, dtype=np.int8))


@pytest.mark.xfail(strict=True, reason="Persian.py: annotated Grid assignment is routed through _GridSetAt, which lacks slice handling")
def test_compiled_whole_grid_slice_scalar_assignment_persian_regression(api):
    grid = api.NewGrid((3, 5), np.int8)

    @api.njit
    def fill(g: api.Grid):
        g[:] = 2

    fill(grid)
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(15, 2, dtype=np.int8))


def test_compiled_whole_grid_slice_read_unannotated(api):
    grid = api.NewGrid((3, 5), np.int32)
    grid[:] = np.arange(15, dtype=np.int32)

    @api.njit
    def read(g):
        return g[:]

    np.testing.assert_array_equal(read(grid), np.arange(15, dtype=np.int32))


@pytest.mark.xfail(strict=True, reason="Annotated Grid read is routed through _GridGetAt, which lacks slice handling")
def test_compiled_whole_grid_slice_read_annotated_regression(api):
    grid = api.NewGrid((3, 5), np.int32)
    grid[:] = np.arange(15, dtype=np.int32)

    @api.njit
    def read(g: api.Grid):
        return g[:]

    np.testing.assert_array_equal(read(grid), np.arange(15, dtype=np.int32))


def test_compiled_whole_grid_slice_augassign_unannotated(api):
    grid = api.NewGrid((3, 5), np.int32)
    grid[:] = 2

    @api.njit
    def increment(g):
        g[:] += 3

    increment(grid)
    np.testing.assert_array_equal(grid[:], np.full(15, 5, dtype=np.int32))


@pytest.mark.xfail(strict=True, reason="Annotated Grid augassign routes slice read/write through _GridGetAt/_GridSetAt, which lack slice handling")
def test_compiled_whole_grid_slice_augassign_annotated_regression(api):
    grid = api.NewGrid((3, 5), np.int32)
    grid[:] = 2

    @api.njit
    def increment(g: api.Grid):
        g[:] += 3

    increment(grid)
    np.testing.assert_array_equal(grid[:], np.full(15, 5, dtype=np.int32))


def test_compiled_partial_tuple_slice_unannotated(api):
    grid = api.NewGrid((4, 5), np.int32)
    grid[:, :] = np.arange(20, dtype=np.int32).reshape(4, 5)

    @api.njit
    def work(g):
        before = g[1:4, 2]
        g[0:2, 1:4] = 9
        return before

    np.testing.assert_array_equal(work(grid), np.array([7, 12, 17], dtype=np.int32))


@pytest.mark.xfail(strict=True, reason="Annotated Grid mixed tuple slice is routed through scalar-only source-aware helpers")
def test_compiled_partial_tuple_slice_annotated_regression(api):
    grid = api.NewGrid((4, 5), np.int32)
    grid[:, :] = np.arange(20, dtype=np.int32).reshape(4, 5)

    @api.njit
    def work(g: api.Grid):
        before = g[1:4, 2]
        g[0:2, 1:4] = 9
        return before

    np.testing.assert_array_equal(work(grid), np.array([7, 12, 17], dtype=np.int32))
