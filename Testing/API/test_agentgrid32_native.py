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
