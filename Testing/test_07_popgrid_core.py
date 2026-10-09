"""PopGrid integer-state and delta-buffer correctness."""
import itertools

import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(8,), (4, 5), (3, 2, 4)])
def test_geometry_matches_c_order(api, shape):
    g = api.NewPopGrid(shape)
    assert g.nDims == len(shape)
    assert len(g) == int(np.prod(shape))
    for coordinates in itertools.product(*(range(n) for n in shape)):
        i = int(np.ravel_multi_index(coordinates, shape, order="C"))
        assert g.ToI(*coordinates) == i
        assert g.ItoX(i) == coordinates[0]
        if len(shape) > 1:
            assert g.ItoY(i) == coordinates[1]
        if len(shape) > 2:
            assert g.ItoZ(i) == coordinates[2]


def test_set_add_update_matches_reference(api):
    shape = (4, 5)
    g = api.NewPopGrid(shape, capacity=100)
    ref = np.zeros(shape, dtype=np.int64)
    g[1, 2] = 10
    g[3, 4] = 20
    ref[1, 2], ref[3, 4] = 10, 20
    np.testing.assert_array_equal(g[:, :], ref)

    deltas = {(1, 2): 5, (3, 4): -7, (0, 0): 9}
    for (x, y), delta in deltas.items():
        g.Add(delta, x, y)
        ref[x, y] += delta

    # Adds are buffered until Update.
    assert g[1, 2] == 10
    assert g[3, 4] == 20
    assert g[0, 0] == 0
    g.Update()
    np.testing.assert_array_equal(g[:, :], ref)
    assert g.GetPop() == int(ref.sum())


def test_multiple_adds_accumulate_before_update(api):
    g = api.NewPopGrid((5,), capacity=1000)
    g[2] = 50
    for delta in (1, 2, -3, 7):
        g.Add(delta, 2)
    assert g[2] == 50
    g.Update()
    assert g[2] == 57
    assert g.GetPop() == 57


def test_reset_clears_values_and_pending_deltas(api):
    g = api.NewPopGrid((5,), capacity=100)
    g[1] = 10
    g.Add(5, 1)
    g.Add(7, 2)
    g.Reset()
    assert g.GetPop() == 0
    np.testing.assert_array_equal(g[:], np.zeros(5, dtype=np.int64))
    g.Update()
    np.testing.assert_array_equal(g[:], np.zeros(5, dtype=np.int64))


def test_all_reports_occupied_indices(api):
    g = api.NewPopGrid((3, 4), capacity=100)
    occupied = [(0, 0), (1, 3), (2, 2)]
    for n, xy in enumerate(occupied, start=1):
        g[xy] = n
    expected = {g.ToI(*xy) for xy in occupied}
    assert set(map(int, g.All())) == expected


@pytest.mark.parametrize("dims,axis,size", [((-5,), "X", 5), ((4, -6), "Y", 6), ((3, 4, -7), "Z", 7)])
def test_wrapping(api, dims, axis, size):
    g = api.NewPopGrid(dims)
    fn = getattr(g, "InWrap" + axis)
    for coordinate in (-size - 1, -1, 0, size - 1, size, 2 * size + 1):
        assert fn(coordinate) == coordinate % size


def test_safe_update_is_atomic_on_underflow(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires caller to preserve population bounds")
    g = api.NewPopGrid((4,), capacity=100)
    g[0] = 5
    g[1] = 6
    before = g[:]
    g.Add(-6, 0)
    g.Add(2, 1)
    with pytest.raises(ValueError):
        g.Update()
    np.testing.assert_array_equal(g[:], before)


def test_safe_update_is_atomic_on_capacity_overflow(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode requires caller to preserve population bounds")
    g = api.NewPopGrid((3,), capacity=10)
    g[0] = 7
    g[1] = 2
    before = g[:]
    g.Add(2, 2)
    with pytest.raises(ValueError):
        g.Update()
    np.testing.assert_array_equal(g[:], before)


def test_safe_rejects_invalid_assignment(api, safe_mode):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise input validation")
    g = api.NewPopGrid((3,), capacity=10)
    for value in (-1, 11, 1.5, float("nan")):
        with pytest.raises((ValueError, OverflowError)):
            g[0] = value
