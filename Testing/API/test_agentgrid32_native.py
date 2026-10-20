"""Native float32 AgentGrid allocation tests; public float32 API is not enabled."""
import ctypes

import numpy as np
import pytest
import NativeCore as core


def test_agentgrid32_native_allocates_float32_properties():
    for dims in ((5,), (3, 4), (-3, 4), (2, 3, 4), ()):
        dimensions = np.array(dims, dtype=np.int32)
        handle = core._ag32_new(dimensions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(dims), 2, 0)
        assert handle
        try:
            expected = int(np.prod(np.abs(dimensions))) if dims else 0
            assert core._ag32_length(handle) == expected
            assert core._ag32_capacity(handle) >= 1
            count = core._ag32_float_property_count(handle)
            assert count == len(dims) + 2
            arr = np.ctypeslib.as_array(core._ag32_float_data(handle), shape=(core._ag32_capacity(handle) * count,))
            assert arr.dtype == np.float32
            arr[0] = np.float32(1.25)
            assert arr[0] == np.float32(1.25)
        finally:
            core._ag32_free(handle)


def test_agentgrid32_rejects_invalid_geometry():
    for dims in ((0,), (2, 0), (-2147483648,), (50000, 50000)):
        dimensions = np.array(dims, dtype=np.int32)
        handle = core._ag32_new(dimensions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(dims), 1, 0)
        assert not handle

def test_agentgrid32_property_operations_are_float32():
    dims = np.array([4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 2, 0)
    assert handle
    try:
        assert core._ag32_new_i(handle, 0) == 0
        assert core._ag32_set_property(handle, 0, 1, np.float32(1.25)) == 1
        assert core._ag32_add_property(handle, 0, 1, np.float32(0.5)) == 1
        assert core._ag32_get_property(handle, 0, 1) == np.float32(1.75)
        assert core._ag32_set_property(handle, 0, 3, np.float32(2)) == 0
        assert core._ag32_add_property(handle, -1, 1, np.float32(1)) == 0
        @core.nb.njit
        def compiled_property(h):
            core._ag32_set_property(h, 0, 2, np.float32(3.5))
            core._ag32_add_property(h, 0, 2, np.float32(0.25))
            return core._ag32_get_property(h, 0, 2)
        assert compiled_property(handle) == np.float32(3.75)
    finally:
        core._ag32_free(handle)

def test_agentgrid32_single_occupancy_lifecycle_and_reuse():
    dims = np.array([4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 0)
    assert handle
    try:
        first = core._ag32_new_i(handle, 2)
        assert first == 0
        assert core._ag32_at_i(handle, 2) == first
        assert core._ag32_population(handle) == 1
        assert core._ag32_new_i(handle, 2) == -1
        assert core._ag32_set_property(handle, first, 1, np.float32(8.0)) == 1
        assert core._ag32_dispose(handle, first) == 1
        assert core._ag32_dispose(handle, first) == 0
        assert core._ag32_at_i(handle, 2) == -1
        assert core._ag32_population(handle) == 0
        assert core._ag32_new_i(handle, 3) == first
        assert core._ag32_get_property(handle, first, 1) == np.float32(0.0)
        @core.nb.njit
        def compiled_cycle(h):
            a = core._ag32_new_i(h, 1)
            n = core._ag32_population(h)
            core._ag32_dispose(h, a)
            return n
        assert compiled_cycle(handle) == 2
    finally:
        core._ag32_free(handle)

def test_agentgrid32_grows_without_losing_properties():
    dims = np.array([1200], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 0)
    assert handle
    try:
        for location in range(1100):
            agent = core._ag32_new_i(handle, location)
            assert agent == location
            assert core._ag32_set_property(handle, agent, 1, np.float32(location + 0.5))
        assert core._ag32_capacity(handle) >= 1100
        assert core._ag32_population(handle) == 1100
        for agent in (0, 499, 999, 1099):
            assert core._ag32_get_property(handle, agent, 1) == np.float32(agent + 0.5)
            assert core._ag32_at_i(handle, agent) == agent
        assert core._ag32_dispose(handle, 1001) == 1
        assert core._ag32_new_i(handle, 1150) == 1001
    finally:
        core._ag32_free(handle)

def test_agentgrid32_square_coordinates_are_float32_centers():
    for dims, location, expected in (
        ((5,), 3, (3.5,)),
        ((3, 4), 6, (1.5, 2.5)),
        ((2, 3, 4), 17, (1.5, 1.5, 1.5)),
    ):
        dimensions = np.array(dims, dtype=np.int32)
        handle = core._ag32_new(dimensions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(dims), 1, 0)
        assert handle
        try:
            agent = core._ag32_new_i(handle, location)
            assert agent == 0
            for axis, value in enumerate(expected):
                assert core._ag32_get_property(handle, agent, axis) == np.float32(value)
            assert core._ag32_get_property(handle, agent, len(dims)) == np.float32(0)
        finally:
            core._ag32_free(handle)

def test_agentgrid32_movement_updates_occupancy_and_coordinates():
    dims = np.array([3, 4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2, 1, 0)
    assert handle
    try:
        first = core._ag32_new_i(handle, 0)
        blocker = core._ag32_new_i(handle, 2)
        assert core._ag32_move_i(handle, first, 2) == 0
        assert core._ag32_at_i(handle, 0) == first
        assert core._ag32_move_i(handle, first, 7) == 1
        assert core._ag32_at_i(handle, 0) == -1
        assert core._ag32_at_i(handle, 7) == first
        assert core._ag32_get_property(handle, first, 0) == np.float32(1.5)
        assert core._ag32_get_property(handle, first, 1) == np.float32(3.5)
        assert core._ag32_move_i(handle, first, 12) == 0
        @core.nb.njit
        def compiled_move(h, agent):
            return core._ag32_move_i(h, agent, 6)
        assert compiled_move(handle, first) == 1
        assert core._ag32_at_i(handle, 6) == first
        assert core._ag32_at_i(handle, 2) == blocker
    finally:
        core._ag32_free(handle)

def test_agentgrid32_stackable_lifecycle_move_and_counts():
    dims = np.array([3], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 1)
    assert handle
    try:
        a = core._ag32_new_i(handle, 0)
        b = core._ag32_new_i(handle, 0)
        c = core._ag32_new_i(handle, 0)
        assert (a, b, c) == (0, 1, 2)
        assert core._ag32_count_i(handle, 0) == 3
        assert core._ag32_at_i(handle, 0) == c
        assert core._ag32_dispose(handle, b) == 1
        assert core._ag32_count_i(handle, 0) == 2
        assert core._ag32_move_i(handle, a, 1) == 1
        assert core._ag32_count_i(handle, 0) == 1
        assert core._ag32_count_i(handle, 1) == 1
        assert core._ag32_move_i(handle, c, 1) == 1
        assert core._ag32_count_i(handle, 1) == 2
        assert core._ag32_at_i(handle, 1) == c
        assert core._ag32_dispose(handle, c) == 1
        assert core._ag32_at_i(handle, 1) == a
        assert core._ag32_count_i(handle, 1) == 1
        assert core._ag32_population(handle) == 1
    finally:
        core._ag32_free(handle)

def test_agentgrid32_stackable_randomized_invariants():
    rng = np.random.default_rng(20261019)
    dims = np.array([8], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 1)
    assert handle
    locations = {}
    try:
        for step in range(1500):
            action = int(rng.integers(3))
            if action == 0 or not locations:
                cell = int(rng.integers(8))
                agent = core._ag32_new_i(handle, cell)
                assert agent >= 0
                locations[agent] = cell
            elif action == 1:
                agent = int(rng.choice(list(locations)))
                cell = int(rng.integers(8))
                assert core._ag32_move_i(handle, agent, cell) == 1
                locations[agent] = cell
            else:
                agent = int(rng.choice(list(locations)))
                assert core._ag32_dispose(handle, agent) == 1
                del locations[agent]
            assert core._ag32_population(handle) == len(locations)
            for cell in range(8):
                assert core._ag32_count_i(handle, cell) == sum(v == cell for v in locations.values())
                head = core._ag32_at_i(handle, cell)
                assert (head == -1) == (core._ag32_count_i(handle, cell) == 0)
                if head != -1:
                    assert locations[head] == cell
    finally:
        core._ag32_free(handle)


def test_agentgrid32_stackable_growth_preserves_chains():
    dims = np.array([3], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 1)
    assert handle
    try:
        for agent in range(1100):
            assert core._ag32_new_i(handle, agent % 3) == agent
        assert core._ag32_capacity(handle) >= 1100
        assert core._ag32_population(handle) == 1100
        for cell in range(3):
            assert core._ag32_count_i(handle, cell) == sum(i % 3 == cell for i in range(1100))
        for agent in (0, 499, 999, 1099):
            assert core._ag32_dispose(handle, agent) == 1
        assert core._ag32_population(handle) == 1096
        assert core._ag32_move_i(handle, 1001, 0) == 1
        assert core._ag32_at_i(handle, 0) == 1001
        assert core._ag32_count_i(handle, 0) == 366
        assert core._ag32_new_i(handle, 2) >= 0
        assert core._ag32_population(handle) == 1097
    finally:
        core._ag32_free(handle)


def test_agentgrid32_compiled_stackable_lifecycle():
    dims = np.array([4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 1)
    assert handle
    try:
        @core.nb.njit
        def compiled_cycle(h):
            a = core._ag32_new_i(h, 0)
            b = core._ag32_new_i(h, 0)
            before = core._ag32_count_i(h, 0)
            moved = core._ag32_move_i(h, a, 2)
            after = core._ag32_count_i(h, 0)
            disposed = core._ag32_dispose(h, b)
            return a, b, before, moved, after, disposed, core._ag32_population(h)
        assert compiled_cycle(handle) == (0, 1, 2, 1, 1, 1, 1)
        assert core._ag32_count_i(handle, 2) == 1
        assert core._ag32_at_i(handle, 2) == 0
    finally:
        core._ag32_free(handle)


def test_agentgrid32_compiled_3d_movement_preserves_float32_centers():
    dims = np.array([3, 4, 5], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3, 1, 0)
    assert handle
    try:
        @core.nb.njit
        def compiled_move(h):
            agent = core._ag32_new_i(h, 0)
            moved = core._ag32_move_i(h, agent, 59)
            return (moved, core._ag32_at_i(h, 0), core._ag32_at_i(h, 59),
                    core._ag32_get_property(h, agent, 0),
                    core._ag32_get_property(h, agent, 1),
                    core._ag32_get_property(h, agent, 2))
        assert compiled_move(handle) == (1, -1, 0, 2.5, 3.5, 4.5)
        assert core._ag32_population(handle) == 1
    finally:
        core._ag32_free(handle)


def test_agentgrid32_point_movement_validates_bounds_and_occupancy():
    dims = np.array([3, 4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2, 1, 0)
    assert handle
    try:
        agent = core._ag32_new_i(handle, 0)
        blocker = core._ag32_new_i(handle, 5)
        assert core._ag32_move_pt(handle, agent, np.float32(1.25), np.float32(1.75), np.float32(0)) == 0
        assert core._ag32_at_i(handle, 0) == agent
        assert core._ag32_move_pt(handle, agent, np.float32(3), np.float32(0), np.float32(0)) == 0
        assert core._ag32_move_pt(handle, agent, np.float32(np.nan), np.float32(0), np.float32(0)) == 0
        @core.nb.njit
        def compiled_move(h, a):
            return core._ag32_move_pt(h, a, np.float32(2.25), np.float32(3.5), np.float32(0))
        assert compiled_move(handle, agent) == 1
        assert core._ag32_at_i(handle, 11) == agent
        assert core._ag32_at_i(handle, 0) == -1
        assert core._ag32_at_i(handle, 5) == blocker
        assert core._ag32_get_property(handle, agent, 0) == np.float32(2.25)
        assert core._ag32_get_property(handle, agent, 1) == np.float32(3.5)
    finally:
        core._ag32_free(handle)


def test_agentgrid32_point_move_within_same_cell_preserves_fractional_position():
    dims = np.array([3, 4, 5], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3, 1, 0)
    assert handle
    try:
        agent = core._ag32_new_i(handle, 0)
        @core.nb.njit
        def move_in_cell(h, a):
            return core._ag32_move_pt(h, a, np.float32(0.125), np.float32(0.75), np.float32(0.875))
        assert move_in_cell(handle, agent) == 1
        assert core._ag32_at_i(handle, 0) == agent
        assert core._ag32_population(handle) == 1
        assert tuple(core._ag32_get_property(handle, agent, axis) for axis in range(3)) == (0.125, 0.75, 0.875)
    finally:
        core._ag32_free(handle)


def test_agentgrid32_stackable_point_move_keeps_occupancy_chains():
    dims = np.array([3], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, 1, 1)
    assert handle
    try:
        a = core._ag32_new_i(handle, 0)
        b = core._ag32_new_i(handle, 0)
        assert core._ag32_move_pt(handle, a, np.float32(1.25), np.float32(0), np.float32(0)) == 1
        assert core._ag32_move_pt(handle, b, np.float32(1.75), np.float32(0), np.float32(0)) == 1
        assert core._ag32_count_i(handle, 0) == 0
        assert core._ag32_count_i(handle, 1) == 2
        assert core._ag32_at_i(handle, 1) == b
        assert core._ag32_get_property(handle, a, 0) == np.float32(1.25)
        assert core._ag32_get_property(handle, b, 0) == np.float32(1.75)
        assert core._ag32_dispose(handle, b) == 1
        assert core._ag32_at_i(handle, 1) == a
        assert core._ag32_count_i(handle, 1) == 1
    finally:
        core._ag32_free(handle)


def test_agentgrid32_point_creation_python_and_compiled():
    dims = np.array([3, 4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2, 1, 0)
    assert handle
    try:
        a = core._ag32_new_pt(handle, np.float32(1.25), np.float32(2.75), np.float32(0))
        assert a == 0
        assert core._ag32_at_i(handle, 6) == a
        assert core._ag32_get_property(handle, a, 0) == np.float32(1.25)
        assert core._ag32_get_property(handle, a, 1) == np.float32(2.75)
        assert core._ag32_new_pt(handle, np.float32(1.5), np.float32(2.5), np.float32(0)) == -1
        assert core._ag32_new_pt(handle, np.float32(np.nan), np.float32(0), np.float32(0)) == -1
        assert core._ag32_population(handle) == 1
        @core.nb.njit
        def compiled_new(h):
            return core._ag32_new_pt(h, np.float32(2.125), np.float32(3.875), np.float32(0))
        b = compiled_new(handle)
        assert b == 1
        assert core._ag32_at_i(handle, 11) == b
        assert core._ag32_get_property(handle, b, 1) == np.float32(3.875)
    finally:
        core._ag32_free(handle)


def test_agentgrid32_point_creation_stackable_and_recycled_ids():
    dims = np.array([2, 3, 4], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3, 1, 1)
    assert handle
    try:
        a = core._ag32_new_pt(handle, np.float32(1.125), np.float32(2.25), np.float32(3.5))
        b = core._ag32_new_pt(handle, np.float32(1.75), np.float32(2.875), np.float32(3.125))
        assert (a, b) == (0, 1)
        assert core._ag32_count_i(handle, 23) == 2
        assert core._ag32_at_i(handle, 23) == b
        assert core._ag32_dispose(handle, b) == 1
        assert core._ag32_at_i(handle, 23) == a
        assert core._ag32_new_pt(handle, np.float32(-0.25), np.float32(1), np.float32(1)) == -1
        assert core._ag32_new_pt(handle, np.float32(2), np.float32(1), np.float32(1)) == -1
        assert core._ag32_population(handle) == 1
        c = core._ag32_new_pt(handle, np.float32(0.375), np.float32(0.625), np.float32(0.875))
        assert c == b
        assert core._ag32_at_i(handle, 0) == c
        assert tuple(core._ag32_get_property(handle, c, axis) for axis in range(3)) == (0.375, 0.625, 0.875)
        assert core._ag32_count_i(handle, 23) == 1
    finally:
        core._ag32_free(handle)


def test_agentgrid32_nonspatial_population_recycles_and_resets_properties():
    dimensions = np.array([], dtype=np.int32)
    handle = core._ag32_new(dimensions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 0, 2, 0)
    assert handle
    try:
        a = core._ag32_new_i(handle, -1)
        b = core._ag32_new_i(handle, -1)
        assert (a, b) == (0, 1)
        assert core._ag32_population(handle) == 2
        assert core._ag32_set_property(handle, a, 0, np.float32(9.5)) == 1
        assert core._ag32_dispose(handle, a) == 1
        assert core._ag32_population(handle) == 1
        c = core._ag32_new_i(handle, -1)
        assert c == a
        assert core._ag32_get_property(handle, c, 0) == np.float32(0)
        assert core._ag32_new_pt(handle, np.float32(0), np.float32(0), np.float32(0)) == -1
        assert core._ag32_population(handle) == 2
    finally:
        core._ag32_free(handle)


def test_agentgrid32_compiled_stackable_point_creation():
    dims = np.array([2, 2], dtype=np.int32)
    handle = core._ag32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2, 1, 1)
    assert handle
    try:
        @core.nb.njit
        def compiled_create(h):
            a = core._ag32_new_pt(h, np.float32(0.125), np.float32(1.25), np.float32(0))
            b = core._ag32_new_pt(h, np.float32(0.875), np.float32(1.75), np.float32(0))
            return a, b, core._ag32_count_i(h, 1), core._ag32_at_i(h, 1)
        assert compiled_create(handle) == (0, 1, 2, 1)
        assert core._ag32_get_property(handle, 0, 0) == np.float32(0.125)
        assert core._ag32_get_property(handle, 1, 1) == np.float32(1.75)
        assert core._ag32_dispose(handle, 1) == 1
        assert core._ag32_at_i(handle, 1) == 0
    finally:
        core._ag32_free(handle)


def test_agentgrid32_internal_factory_manages_native_lifetime():
    with core._NewAgentGrid32((3, 4), numAgentProps=1, isStackable=False) as grid:
        assert grid.GetPop() == 0
        a = grid.NewAgentPT(1.25, 2.75)
        assert a == 0
        assert grid.GetPop() == 1
        assert grid.GetProperty(a, 0) == np.float32(0)
        assert grid.SetProperty(a, 0, 7.5)
        assert grid.GetProperty(a, 0) == np.float32(7.5)
        assert grid.MovePT(a, 2.125, 3.5)
        assert grid.GetProperty(a, 0) == np.float32(7.5)
        assert core._ag32_get_property(grid._ptr, a, 0) == np.float32(2.125)
        assert core._ag32_get_property(grid._ptr, a, 1) == np.float32(3.5)
        assert grid.Dispose(a)
        assert grid.GetPop() == 0
    assert grid._ptr == 0
    grid.Close()


def test_agentgrid32_internal_wrapper_stackable_occupancy_and_property_add():
    with core._NewAgentGrid32((3,), numAgentProps=1, isStackable=True) as grid:
        a = grid.NewAgentPT(1.125)
        b = grid.NewAgentPT(1.75)
        assert grid.CountI(1) == 2
        assert grid.AtI(1) == b
        assert grid.SetProperty(a, 0, 1.25)
        assert grid.AddProperty(a, 0, 0.5)
        assert grid.GetProperty(a, 0) == np.float32(1.75)
        assert grid.MoveI(b, 2)
        assert grid.CountI(1) == 1
        assert grid.AtI(1) == a
        assert grid.CountI(2) == 1


def test_agentgrid32_internal_factory_rejects_invalid_options():
    import pytest
    for props in (-1, 1.5, True, 2**31):
        with pytest.raises(ValueError, match="numAgentProps"):
            core._NewAgentGrid32((2, 2), numAgentProps=props)
    for stackable in (-1, 2, 0.5, "yes"):
        with pytest.raises(ValueError, match="isStackable"):
            core._NewAgentGrid32((2, 2), isStackable=stackable)


def test_agentgrid32_internal_factory_geometry():
    with core._NewAgentGrid32((-3, 4), numAgentProps=1) as grid:
        assert grid.nDims == 2
        assert grid.dimensions == (3, 4)
        assert len(grid) == 12
    with core._NewAgentGrid32((), numAgentProps=1) as grid:
        assert grid.nDims == 0
        assert len(grid) == 0


def test_float32_agentgrid_square_construction_and_bounds():
    with core._NewAgentGrid32((3, 4), numAgentProps=1) as grid:
        agent = grid.NewAgentSQ(1, 2)
        assert agent >= 0
        assert grid.AtI(6) == agent
        assert grid.GetPop() == 1
        with pytest.raises(ValueError):
            grid.NewAgent(2)
        for coords in [(-1, 2), (3, 0), (0, 4), (1.5, 2)]:
            with pytest.raises(IndexError):
                grid.NewAgentSQ(*coords)
        with pytest.raises(ValueError):
            grid.NewAgentSQ(1)
    with core._NewAgentGrid32((4,), numAgentProps=1) as grid:
        agent = grid.NewAgent(2)
        assert grid.AtI(2) == agent


def test_float32_agentgrid_square_movement_and_bounds():
    with core._NewAgentGrid32((3, 4), numAgentProps=1) as grid:
        agent = grid.NewAgentSQ(1, 2)
        assert grid.MoveSQ(agent, 2, 1)
        assert grid.AtI(9) == agent
        assert grid.AtI(6) != agent
        with pytest.raises(ValueError):
            grid.Move(agent, 1)
        for coords in [(-1, 2), (3, 0), (0, 4), (1.5, 2)]:
            with pytest.raises(IndexError):
                grid.MoveSQ(agent, *coords)
        assert grid.AtI(9) == agent
    with core._NewAgentGrid32((4,), numAgentProps=1) as grid:
        agent = grid.NewAgent(1)
        assert grid.Move(agent, 3)
        assert grid.AtI(3) == agent


def test_float32_agentgrid_square_occupancy_and_stacking():
    with core._NewAgentGrid32((3, 4), isStackable=True) as grid:
        first = grid.NewAgentSQ(1, 2)
        second = grid.NewAgentSQ(1, 2)
        assert grid.CountSQ(1, 2) == 2
        assert grid.AtSQ(1, 2) in (first, second)
        assert grid.CountSQ(0, 0) == 0
        assert grid.MoveSQ(second, 2, 1)
        assert grid.CountSQ(1, 2) == 1
        assert grid.CountSQ(2, 1) == 1
        assert grid.Dispose(first)
        assert grid.CountSQ(1, 2) == 0
        with pytest.raises(IndexError):
            grid.AtSQ(3, 0)


def test_float32_agentgrid_linear_index_validation():
    with core._NewAgentGrid32((3, 4), isStackable=True) as grid:
        agent = grid.NewAgentI(6)
        for invalid in (-1, 12, 1.5, float("nan"), True):
            for operation in (lambda: grid.NewAgentI(invalid),
                              lambda: grid.MoveI(agent, invalid),
                              lambda: grid.AtI(invalid),
                              lambda: grid.CountI(invalid)):
                with pytest.raises((ValueError, IndexError)):
                    operation()
        assert grid.AtI(6) == agent
        assert grid.CountI(6) == 1


def test_float32_agentgrid_point_creation_and_movement():
    with core._NewAgentGrid32((3, 4), numAgentProps=1) as grid:
        agent = grid.NewAgentPT(1.25, 2.5)
        assert grid.AtSQ(1, 2) == agent
        assert grid.MovePT(agent, 2.25, 1.5)
        assert grid.AtSQ(2, 1) == agent
        for coords in [(-0.5, 1), (3, 1), (float("nan"), 1), (float("inf"), 1), (1, 4)]:
            with pytest.raises((ValueError, IndexError)):
                grid.NewAgentPT(*coords)
            with pytest.raises((ValueError, IndexError)):
                grid.MovePT(agent, *coords)
        with pytest.raises(ValueError):
            grid.NewAgentPT(1, 2, 1)
        assert grid.AtSQ(2, 1) == agent


def test_float32_agentgrid_property_validation():
    with core._NewAgentGrid32((3,), numAgentProps=2) as grid:
        agent = grid.NewAgent(1)
        assert grid.SetProperty(agent, 0, 1.25)
        assert grid.AddProperty(agent, 0, 0.5)
        assert grid.GetProperty(agent, 0) == pytest.approx(1.75)
        for invalid in (-1, 2, 0.5, True):
            with pytest.raises(IndexError):
                grid.GetProperty(agent, invalid)
            with pytest.raises(IndexError):
                grid.SetProperty(agent, invalid, 1)
            with pytest.raises(IndexError):
                grid.AddProperty(agent, invalid, 1)
        for invalid in (float("nan"), float("inf"), 1e100):
            with pytest.raises(ValueError):
                grid.SetProperty(agent, 0, invalid)
            with pytest.raises(ValueError):
                grid.AddProperty(agent, 0, invalid)
        assert grid.GetProperty(agent, 0) == pytest.approx(1.75)


def test_float32_agentgrid_point_bounds_preserve_state():
    with core._NewAgentGrid32((3, 4), numAgentProps=1) as grid:
        agent = grid.NewAgentPT(1.25, 2.5)
        assert grid.SetProperty(agent, 0, 6.5)
        for x, y in ((-0.1, 1), (3, 1), (1, -0.1), (1, 4),
                     (float("nan"), 1), (float("inf"), 1), (1, float("-inf"))):
            with pytest.raises((ValueError, IndexError)):
                grid.MovePT(agent, x, y)
            with pytest.raises((ValueError, IndexError)):
                grid.NewAgentPT(x, y)
            assert grid.AtSQ(1, 2) == agent
            assert grid.GetPop() == 1
            assert grid.GetProperty(agent, 0) == np.float32(6.5)
        assert grid.MovePT(agent, 2.25, 3.5)
        assert grid.AtSQ(1, 2) == -1
        assert grid.AtSQ(2, 3) == agent


def test_float32_agentgrid_dead_agent_properties_are_inaccessible():
    with core._NewAgentGrid32((3,), numAgentProps=1) as grid:
        agent = grid.NewAgent(1)
        assert grid.SetProperty(agent, 0, 3.0)
        assert grid.Dispose(agent)
        assert not grid.SetProperty(agent, 0, 4.0)
        assert not grid.AddProperty(agent, 0, 1.0)
        assert np.isnan(grid.GetProperty(agent, 0))
        assert not grid.SetProperty(agent + 100, 0, 4.0)


def test_float32_agentgrid_dispose_alive_list_and_reuse():
    with core._NewAgentGrid32((8,), numAgentProps=1) as grid:
        agents = [grid.NewAgent(i) for i in range(5)]
        assert grid.GetPop() == 5
        assert grid.Dispose(agents[1])
        assert grid.GetPop() == 4
        assert grid.Dispose(agents[4])
        assert grid.GetPop() == 3
        assert not grid.Dispose(agents[1])
        reused = grid.NewAgent(6)
        assert reused in (agents[1], agents[4])
        assert grid.GetPop() == 4
        assert grid.AtI(6) == reused
        for agent in (agents[0], agents[2], agents[3], reused):
            assert grid.Dispose(agent)
        assert grid.GetPop() == 0
        assert grid.NewAgent(7) >= 0
        assert grid.GetPop() == 1


def test_float32_agentgrid_property_addition_overflow_is_atomic():
    with core._NewAgentGrid32((3,), numAgentProps=1) as grid:
        agent = grid.NewAgent(1)
        assert grid.SetProperty(agent, 0, 3e38)
        with pytest.raises(ValueError, match="overflow"):
            grid.AddProperty(agent, 0, 1e38)
        assert grid.GetProperty(agent, 0) == pytest.approx(3e38, rel=1e-6)
        assert grid.AddProperty(agent, 0, -1e38)
        assert grid.GetProperty(agent, 0) == pytest.approx(2e38, rel=1e-6)


def test_safe_public_agentgrid_requires_all_square_coordinates():
    original = core._UseFastMode
    try:
        for fast in (False, True):
            core._UseFastMode = lambda: fast
            for dims in ((3, 4), (2, 3, 4)):
                grid = core.NewAgentGrid(dims, numAgentProps=1)
                with pytest.raises(ValueError, match="all coordinates"):
                    grid.NewAgentSQ(1)
                if len(dims) == 3:
                    with pytest.raises(ValueError, match="all coordinates"):
                        grid.NewAgentSQ(1, 2)
                if len(dims) == 2:
                    assert grid.NewAgentSQ(1, 2) >= 0
                else:
                    assert grid.NewAgentSQ(1, 2, 1) >= 0
    finally:
        core._UseFastMode = original


def test_agentgrid32_native_property_indexing_matches_explicit_access():
    with core._NewAgentGrid32((4,), numAgentProps=2) as grid:
        agent = grid.NewAgent(1)
        assert agent >= 0
        grid[agent, 0] = 1.25
        grid[agent, 1] = -2.5
        assert grid[agent, 0] == pytest.approx(1.25)
        assert grid.GetProperty(agent, 1) == pytest.approx(-2.5)
        grid.AddProperty(agent, 0, 0.5)
        assert grid[agent, 0] == pytest.approx(1.75)
        with pytest.raises(IndexError):
            grid[agent, 2] = 0.0
        with pytest.raises(TypeError):
            grid[agent] = 1.0
        with pytest.raises(ValueError):
            grid[agent, 0] = float("inf")


@pytest.mark.parametrize("dims", [(5,), (3, 4), (-3, 4), (2, 3, 4), ()])
def test_public_float32_agentgrid_python_constructor_and_properties(dims):
    with core.NewAgentGrid(dims, numAgentProps=2, dtype=np.float32) as grid:
        assert grid.nDims == len(dims)
        for axis, name in enumerate(("xDim", "yDim", "zDim")):
            if axis < len(dims):
                assert getattr(grid, name) == abs(dims[axis])
            else:
                with pytest.raises(ValueError):
                    getattr(grid, name)
        assert grid.GetPop() == 0
        if dims:
            agent = grid.NewAgentSQ(*([0] * len(dims)))
            assert agent >= 0
            grid[agent, 0] = 1.25
            grid[agent, 1] = 2.5
            assert grid[agent, 0] == pytest.approx(1.25)
            assert grid[agent, 1] == pytest.approx(2.5)
            assert grid.GetPop() == 1


def test_public_agentgrid_dtype_validation():
    with pytest.raises(ValueError, match="float32 or float64"):
        core.NewAgentGrid((4,), dtype=np.int32)


@pytest.mark.parametrize("dims", [(5,), (3, 4), (-3, 4), (2, 3, 4)])
def test_public_float32_agentgrid_coordinate_roundtrip_and_wrap_flags(dims):
    with core.NewAgentGrid(dims, dtype=np.float32) as grid:
        assert (grid.wrapX, grid.wrapY, grid.wrapZ) == tuple(
            dims[axis] < 0 if axis < len(dims) else False for axis in range(3)
        )
        last = tuple(abs(d) - 1 for d in dims)
        index = grid.ToI(*last)
        assert index == len(grid) - 1
        assert grid.ItoX(index) == last[0]
        if len(dims) >= 2:
            assert grid.ItoY(index) == last[1]
        else:
            with pytest.raises(ValueError):
                grid.ItoY(index)
        if len(dims) >= 3:
            assert grid.ItoZ(index) == last[2]
        else:
            with pytest.raises(ValueError):
                grid.ItoZ(index)
