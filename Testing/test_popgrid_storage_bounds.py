"""Regression tests for PopGrid explicit dtype storage widths."""
import os
import pytest
import numpy as np
import PythonAutomataLibrary as pal

SAFE = os.environ.get("PAL_TEST_MODE", "safe").lower() != "fast"


def test_uint8_storage_accepts_values_above_old_capacity_hint():
    grid = pal.NewPopGrid((2,), dtype=np.uint8)  # uint8: 0..255
    grid[0] = 150
    assert grid[0] == 150
    grid.Add(50, 0)
    grid.Update()
    assert grid[0] == 200
    assert grid.GetPop() == 200


@pytest.mark.skipif(not SAFE, reason="Fast mode intentionally skips overflow checks")
def test_direct_write_rejects_storage_overflow():
    grid = pal.NewPopGrid((2,), dtype=np.uint8)
    grid[0] = 200
    with pytest.raises(ValueError):
        grid[0] = 256
    assert grid[0] == 200
    assert grid.GetPop() == 200


@pytest.mark.skipif(not SAFE, reason="Fast mode intentionally skips overflow checks")
def test_update_overflow_is_atomic_and_preserves_deltas():
    grid = pal.NewPopGrid((2,), dtype=np.uint8)
    grid[0] = 250
    grid.Add(10, 0)
    grid.Add(5, 1)
    with pytest.raises(ValueError):
        grid.Update()
    assert grid[0] == 250
    assert grid[1] == 0
    assert grid.GetPop() == 250
    grid.Add(-10, 0)  # cancel the invalid pending change
    grid.Update()
    assert grid[0] == 250
    assert grid[1] == 5
    assert grid.GetPop() == 255


@pytest.mark.skipif(not SAFE, reason="Fast mode intentionally skips overflow checks")
def test_final_total_checked_without_intermediate_overflow():
    grid = pal.NewPopGrid((2,), dtype=np.int64)  # int64
    grid[0] = 2**63 - 1
    grid.Add(-(2**63 - 1), 0)
    grid.Add(1, 1)
    grid.Update()
    assert grid[0] == 0
    assert grid[1] == 1
    assert grid.GetPop() == 1

@pal.njit
def _compiled_write_and_update(grid):
    grid[0] = 150
    grid.Add(50, 0)
    grid.Update()
    return grid[0]


def test_compiled_path_respects_uint8_storage():
    grid = pal.NewPopGrid((2,), dtype=np.uint8)
    assert _compiled_write_and_update(grid) == 200
    assert grid.GetPop() == 200


@pal.njit
def _compiled_overflow(grid):
    grid[0] = 256


@pytest.mark.skipif(not SAFE, reason="Fast mode intentionally skips overflow checks")
def test_compiled_path_rejects_storage_overflow():
    grid = pal.NewPopGrid((2,), dtype=np.uint8)
    with pytest.raises(ValueError):
        _compiled_overflow(grid)
    assert grid[0] == 0

@pytest.mark.parametrize("dtype,maximum", [(np.uint8, 255), (np.uint16, 65535), (np.uint32, 2**32 - 1)])
def test_storage_width_boundaries(dtype, maximum):
    grid = pal.NewPopGrid((1,), dtype=dtype)
    grid[0] = maximum
    assert grid[0] == maximum
    assert grid.GetPop() == maximum
    if SAFE:
        with pytest.raises(ValueError):
            grid[0] = maximum + 1
        assert grid[0] == maximum
