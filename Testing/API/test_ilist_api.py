"""IList API parity and composition contracts."""
import numpy as np
import pytest


def test_mutators_are_chainable_in_python(api):
    q = api.NewIList()
    assert q.Append(3) is not None
    assert q.Append(7).Clear() is not None
    assert q.Append(5).Shuffle() is not None
    assert list(map(int, q.All())) == [5]


def test_iter_is_live_view_while_all_is_copy(api):
    q = api.NewIList()
    for value in (2, 4, 6):
        q.Append(value)
    live = q.Iter()
    copy = q.All()
    live[1] = 99
    assert q[1] == 99
    assert copy[1] == 4


def test_compiled_append_clear_index_and_iter(api):
    q = api.NewIList()

    @api.njit
    def work(out):
        out.Append(4)
        out.Append(9)
        before = out[1]
        total = 0
        for value in out.Iter():
            total += value
        out.Clear()
        out.Append(total)
        return before, len(out), out[0]

    assert work(q) == (9, 1, 13)
    assert q[0] == 13


def test_compiled_all_returns_detached_copy(api):
    q = api.NewIList()
    q.Append(1).Append(2)

    @api.njit
    def copy_and_change(out):
        values = out.All()
        values[0] = 88
        return values[0], out[0]

    assert copy_and_change(q) == (88, 1)


@pytest.mark.parametrize("value", [-1, 2**31, 1.5, np.nan, np.inf])
def test_compiled_safe_append_validation(api, safe_mode, value):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits IList value validation")
    q = api.NewIList()

    @api.njit
    def append(out, x):
        out.Append(x)

    with pytest.raises(ValueError):
        append(q, value)
