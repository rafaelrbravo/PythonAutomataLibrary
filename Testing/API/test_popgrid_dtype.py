"""PopGrid dtype selection contract."""
import numpy as np
import pytest
import PythonAutomataLibrary as pal


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int64])
def test_popgrid_explicit_dtype(dtype):
    grid = pal.NewPopGrid((4, 4), dtype=dtype)
    assert grid is not None


def test_popgrid_default_dtype():
    assert pal.NewPopGrid((4, 4)) is not None


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint64, np.int8, np.int16, np.int32])
def test_popgrid_unsupported_dtype_rejected(dtype):
    with pytest.raises(ValueError):
        pal.NewPopGrid((4, 4), dtype=dtype)


@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.uint16, np.uint32])
def test_popgrid_dtype_range(dtype):
    grid = pal.NewPopGrid((3,), dtype=dtype)
    maximum = min(np.iinfo(dtype).max, 1000)
    grid[0] = maximum
    assert grid[0] == maximum
    if np.iinfo(dtype).max < 1000:
        with pytest.raises((ValueError, OverflowError)):
            grid[0] = np.iinfo(dtype).max + 1
