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
    def fill(g: api.Grid):
        g[:] = 2

    fill(grid)
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(15, 2, dtype=np.int8))


@pytest.mark.xfail(strict=True, reason="Persian.py: annotated Grid assignment is routed through _GridSetAt, which lacks slice handling")
def test_compiled_whole_grid_slice_scalar_assignment_persian_regression(api):
    grid = api.NewGrid((3, 5), np.int8)

    @api.njit
    def fill(g):
        g[:] = 2

    fill(grid)
    np.testing.assert_array_equal(np.asarray(grid[:]), np.full(15, 2, dtype=np.int8))
