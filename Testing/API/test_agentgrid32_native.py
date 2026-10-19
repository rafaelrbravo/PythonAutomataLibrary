"""Native float32 AgentGrid allocation tests; public float32 API is not enabled."""
import ctypes

import numpy as np
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
