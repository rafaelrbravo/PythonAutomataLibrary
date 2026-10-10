"""PopGrid API contracts beyond the core algorithmic tests."""
import numpy as np
import pytest


@pytest.mark.parametrize("shape,key,value", [
    ((7,), 3, 11),
    ((4, 5), (2, 3), 12),
    ((3, 4, 5), (1, 2, 4), 13),
])
def test_scalar_index_and_coordinate_assignment_roundtrip(api, shape, key, value):
    g = api.NewPopGrid(shape, capacity=100)
    g[key] = value
    assert g[key] == value
    assert g.GetPop() == value


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5)])
def test_slice_assignment_and_copy_semantics(api, shape):
    g = api.NewPopGrid(shape, capacity=100)
    values = np.arange(len(g), dtype=np.int64).reshape(shape)
    key = slice(None) if len(shape) == 1 else tuple(slice(None) for _ in shape)
    g[key] = values
    got = g[key]
    np.testing.assert_array_equal(got, values)
    got.flat[0] = 99
    scalar_key = 0 if len(shape) == 1 else tuple(0 for _ in shape)
    assert g[scalar_key] == 0

def test_pending_add_does_not_change_all_until_update(api):
    g = api.NewPopGrid((5,), capacity=100)
    g.Add(7, 2)
    assert len(g.All()) == 0
    assert g.GetPop() == 0
    g.Update()
    assert list(map(int, g.All())) == [2]
    assert g.GetPop() == 7


def test_reset_is_compilable_and_clears_pending_delta(api):
    g = api.NewPopGrid((5,), capacity=100)
    g[1] = 8

    @api.njit
    def work(grid):
        grid.Add(3, 1)
        grid.Add(4, 2)
        grid.Reset()
        grid.Update()
        return grid.GetPop()

    assert work(g) == 0
    np.testing.assert_array_equal(g[:], np.zeros(5, dtype=np.int64))


def test_compiled_keyword_add_all_dimensions(api):
    grids = [api.NewPopGrid((5,), capacity=100), api.NewPopGrid((4, 5), capacity=100), api.NewPopGrid((3, 4, 5), capacity=100)]

    @api.njit
    def one(g):
        g.Add(value=2, x=3)
        g.Update()

    @api.njit
    def two(g):
        g.Add(value=3, x=2, y=4)
        g.Update()

    @api.njit
    def three(g):
        g.Add(value=4, x=1, y=2, z=3)
        g.Update()

    one(grids[0]); two(grids[1]); three(grids[2])
    assert grids[0][3] == 2
    assert grids[1][2, 4] == 3
    assert grids[2][1, 2, 3] == 4


def test_safe_delta_int64_overflow_is_atomic(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits PopGrid overflow validation")
    g = api.NewPopGrid((2,), capacity=2**63 - 1)
    g.Add(2**63 - 1, 0)
    with pytest.raises(OverflowError):
        g.Add(1, 0)
    g.Reset()
    assert g.GetPop() == 0


@pytest.mark.parametrize("method", ["InWrapY", "InWrapZ"])
def test_safe_missing_dimension_wrap_methods_rejected(api, safe_mode, method):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits dimensional validation")
    g = api.NewPopGrid((5,))
    with pytest.raises(ValueError):
        getattr(g, method)(0)
