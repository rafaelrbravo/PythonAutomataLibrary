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


def test_python_njit_ilist_mutation_and_copy_parity(api):
    """Compare outputs and final list contents after identical mutations."""
    py = api.NewIList()
    jit = api.NewIList()

    def python_work(q):
        q.Append(4)
        q.Append(9)
        before = q[1]
        total = 0
        for value in q.Iter():
            total += value
        detached = q.All()
        detached[0] = 77
        still_original = q[0]
        q.Clear()
        q.Append(total)
        return before, still_original, len(q), q[0]

    @api.njit
    def compiled_work(q):
        q.Append(4)
        q.Append(9)
        before = q[1]
        total = 0
        for value in q.Iter():
            total += value
        detached = q.All()
        detached[0] = 77
        still_original = q[0]
        q.Clear()
        q.Append(total)
        return before, still_original, len(q), q[0]

    assert compiled_work(jit) == python_work(py)
    np.testing.assert_array_equal(jit.All(), py.All())


@pytest.mark.parametrize("value", [-1, 2**31, 1.5, np.nan, np.inf])
def test_python_njit_ilist_invalid_append_atomic_parity(api, safe_mode, value):
    """Compare safe-mode failure classes and unchanged contents after Append."""
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits IList value validation")
    py = api.NewIList()
    jit = api.NewIList()
    for q in (py, jit):
        q.Append(3)
        q.Append(7)

    @api.njit
    def compiled_append(q, x):
        q.Append(x)

    with pytest.raises(ValueError) as python_error:
        py.Append(value)
    with pytest.raises(ValueError) as compiled_error:
        compiled_append(jit, value)
    assert type(python_error.value) is type(compiled_error.value)
    np.testing.assert_array_equal(py.All(), np.array([3, 7]))
    np.testing.assert_array_equal(jit.All(), np.array([3, 7]))


def test_python_njit_ilist_iter_live_and_all_copy_parity(api):
    """Mutating Iter must affect the list, while mutating All must not."""
    py = api.NewIList()
    jit = api.NewIList()

    def python_work(q):
        q.Append(2)
        q.Append(4)
        q.Append(6)
        live = q.Iter()
        detached = q.All()
        live[1] = 19
        detached[0] = 88
        return q[0], q[1], detached[0], len(q)

    @api.njit
    def compiled_work(q):
        q.Append(2)
        q.Append(4)
        q.Append(6)
        live = q.Iter()
        detached = q.All()
        live[1] = 19
        detached[0] = 88
        return q[0], q[1], detached[0], len(q)

    assert compiled_work(jit) == python_work(py) == (2, 19, 88, 3)
    np.testing.assert_array_equal(jit.All(), py.All())


def test_python_njit_ilist_clear_and_reuse_parity(api):
    """Clear must reset logical length and permit subsequent appends."""
    py = api.NewIList()
    jit = api.NewIList()

    def python_work(q):
        q.Append(3)
        q.Append(5)
        before = len(q)
        q.Clear()
        empty = len(q)
        q.Append(11)
        q.Append(13)
        return before, empty, len(q), q[0], q[1]

    @api.njit
    def compiled_work(q):
        q.Append(3)
        q.Append(5)
        before = len(q)
        q.Clear()
        empty = len(q)
        q.Append(11)
        q.Append(13)
        return before, empty, len(q), q[0], q[1]

    assert compiled_work(jit) == python_work(py) == (2, 0, 2, 11, 13)
    np.testing.assert_array_equal(jit.All(), py.All())


def test_python_njit_ilist_index_after_clear_parity(api):
    py = api.NewIList()
    jit = api.NewIList()

    def python_work(q):
        q.Append(10)
        q.Append(20)
        q.Clear()
        q.Append(30)
        return len(q), q[0]

    @api.njit
    def compiled_work(q):
        q.Append(10)
        q.Append(20)
        q.Clear()
        q.Append(30)
        return len(q), q[0]

    assert compiled_work(jit) == python_work(py) == (1, 30)
    np.testing.assert_array_equal(jit.All(), py.All())


def test_python_njit_ilist_repeated_clear_parity(api):
    py = api.NewIList()
    jit = api.NewIList()

    def python_work(q):
        q.Clear()
        q.Append(7)
        q.Clear()
        q.Clear()
        q.Append(12)
        q.Append(14)
        return len(q), q[0], q[1]

    @api.njit
    def compiled_work(q):
        q.Clear()
        q.Append(7)
        q.Clear()
        q.Clear()
        q.Append(12)
        q.Append(14)
        return len(q), q[0], q[1]

    assert compiled_work(jit) == python_work(py) == (2, 12, 14)
    np.testing.assert_array_equal(jit.All(), py.All())
