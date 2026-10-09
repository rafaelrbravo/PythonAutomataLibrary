"""IList storage and randomization invariants, without depending on RNG sequence."""
from collections import Counter

import numpy as np
import pytest


def test_append_index_and_all(api):
    q = api.NewIList()
    values = [7, 0, 11, 7, 2, 999]
    for value in values:
        q.Append(value)
    assert len(q) == len(values)
    assert [int(q[i]) for i in range(len(q))] == values
    np.testing.assert_array_equal(q.All(), np.asarray(values, dtype=np.int32))


def test_all_is_a_snapshot(api):
    q = api.NewIList()
    q.Append(2).Append(5)
    snapshot = q.All()
    snapshot[0] = 999
    assert q[0] == 2
    q.Append(8)
    assert len(snapshot) == 2


def test_shuffle_is_a_permutation(api):
    q = api.NewIList()
    values = [1, 1, 2, 3, 5, 8, 13]
    for value in values:
        q.Append(value)
    for _ in range(8):
        q.Shuffle()
        assert Counter(map(int, q.All())) == Counter(values)


def test_random_returns_member(api):
    q = api.NewIList()
    for value in [4, 9, 15]:
        q.Append(value)
    assert {int(q.Random()) for _ in range(200)} <= {4, 9, 15}


def test_clear_and_reuse(api):
    q = api.NewIList()
    for value in range(40):
        q.Append(value)
    q.Clear()
    assert len(q) == 0
    q.Append(17)
    assert len(q) == 1
    assert q[0] == 17


@pytest.mark.parametrize("value", [-1, 2**31, 1.5, float("nan")])
def test_safe_append_validation(api, safe_mode, value):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise input validation")
    q = api.NewIList()
    with pytest.raises((ValueError, OverflowError)):
        q.Append(value)


def test_safe_empty_random_and_bounds(api, safe_mode):
    if not safe_mode:
        pytest.skip("Unchecked mode intentionally does not promise input validation")
    q = api.NewIList()
    with pytest.raises(ValueError):
        q.Random()
    q.Append(2)
    with pytest.raises(IndexError):
        _ = q[1]
