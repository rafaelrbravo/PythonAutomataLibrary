"""Direct native float32 PDE storage and diffusion tests until Python bindings are wired."""
import ctypes

import numpy as np
import pytest
import NativeCore as core


def _native(name, restype, *argtypes):
    fn = getattr(core._LIB, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def test_native_pdegrid32_storage_and_update():
    ptr = ctypes.c_uint64
    i32 = ctypes.c_int32
    f32 = ctypes.c_float
    dims = (i32 * 2)(4, 4)
    new = _native("pal_pd32_new", ptr, ctypes.POINTER(i32), i32)
    free = _native("pal_pd32_free", None, ptr)
    get = _native("pal_pd32_geti", f32, ptr, i32)
    set_value = _native("pal_pd32_seti", i32, ptr, i32, f32)
    add = _native("pal_pd32_addi", None, ptr, i32, f32)
    update = _native("pal_pd32_update", None, ptr)
    field_ptr = _native("pal_pd32_field_data", ctypes.POINTER(f32), ptr)
    handle = new(dims, 2)
    assert handle
    try:
        assert set_value(handle, 5, f32(1.25)) == 1
        add(handle, 5, f32(0.5))
        update(handle)
        assert get(handle, 5) == pytest.approx(1.75)
        assert np.ctypeslib.as_array(field_ptr(handle), shape=(16,)).dtype == np.float32
    finally:
        free(handle)


@pytest.mark.parametrize("dimensions", [(5,), (5, 5), (5, 5, 5), (-5,)])
def test_native_pdegrid32_diffusion_conserves_mass(dimensions):
    ptr = ctypes.c_uint64
    i32 = ctypes.c_int32
    f32 = ctypes.c_float
    new = _native("pal_pd32_new", ptr, ctypes.POINTER(i32), i32)
    free = _native("pal_pd32_free", None, ptr)
    set_value = _native("pal_pd32_seti", i32, ptr, i32, f32)
    diffuse = _native("pal_pd32_diffuse", i32, ptr, ctypes.c_double)
    update = _native("pal_pd32_update", None, ptr)
    field_ptr = _native("pal_pd32_field_data", ctypes.POINTER(f32), ptr)
    dims = (i32 * len(dimensions))(*dimensions)
    handle = new(dims, len(dimensions))
    assert handle
    try:
        field = np.ctypeslib.as_array(field_ptr(handle), shape=(5 ** len(dimensions),))
        assert set_value(handle, len(field) // 2, f32(1.0)) == 1
        assert diffuse(handle, 0.1) == 1
        update(handle)
        assert field.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-6)
        assert np.all(field >= 0.0)
    finally:
        free(handle)

def test_native_pdegrid32_unstable_diffusion_does_not_mutate_deltas():
    ptr = ctypes.c_uint64
    i32 = ctypes.c_int32
    f32 = ctypes.c_float
    new = _native("pal_pd32_new", ptr, ctypes.POINTER(i32), i32)
    free = _native("pal_pd32_free", None, ptr)
    set_value = _native("pal_pd32_seti", i32, ptr, i32, f32)
    diffuse = _native("pal_pd32_diffuse", i32, ptr, ctypes.c_double)
    update = _native("pal_pd32_update", None, ptr)
    field_ptr = _native("pal_pd32_field_data", ctypes.POINTER(f32), ptr)
    handle = new((i32 * 2)(5, 5), 2)
    assert handle
    try:
        field = np.ctypeslib.as_array(field_ptr(handle), shape=(25,))
        assert set_value(handle, 12, f32(1.0)) == 1
        assert diffuse(handle, 0.3) == 0  # 2 * 0.3 > explicit stability limit
        update(handle)
        assert field[12] == 1.0
        assert np.count_nonzero(field) == 1
    finally:
        free(handle)

def test_bound_pdegrid32_functions():
    dims = np.array([4, 4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2)
    assert handle
    try:
        assert core._pd32_len(handle) == 16
        assert core._pd32_dim(handle, 0) == 4
        assert core._pd32_toi(handle, 1, 1, 0) == 5
        assert core._pd32_seti(handle, 5, ctypes.c_float(1.0)) == 1
        assert core._pd32_diffuse(handle, 0.1) == 1
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 5) == pytest.approx(0.6, abs=1e-6)
    finally:
        core._pd32_free(handle)

def test_compiled_pdegrid32_native_bindings():
    @core.nb.njit
    def run(handle):
        if core._pd32_seti(handle, 5, np.float32(1.0)) != 1:
            return np.float32(-1.0)
        if core._pd32_diffuse(handle, 0.1) != 1:
            return np.float32(-2.0)
        core._pd32_update(handle)
        return core._pd32_geti(handle, 5)

    dims = np.array([4, 4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2)
    assert handle
    try:
        assert run(handle) == pytest.approx(0.6, abs=1e-6)
    finally:
        core._pd32_free(handle)

def test_compiled_pdegrid32_coordinate_access():
    @core.nb.njit
    def run(handle):
        assert core._pd32_set(handle, 1, 2, 0, np.float32(1.25)) == 1
        assert core._pd32_add(handle, 1, 2, 0, np.float32(0.5)) == 1
        core._pd32_update(handle)
        return core._pd32_get(handle, 1, 2, 0)

    dims = np.array([4, 4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2)
    assert handle
    try:
        assert run(handle) == pytest.approx(1.75)
        assert core._pd32_set(handle, 5, 0, 0, ctypes.c_float(2.0)) == 0
    finally:
        core._pd32_free(handle)

def test_pdegrid32_linear_slices_and_atomic_validation():
    dims = np.array([6], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
    assert handle
    try:
        values = np.array([1.25, 2.5, 3.75], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        assert core._pd32_linear_set_array(handle, 1, 4, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 1
        core._pd32_linear_get(handle, 1, 4, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        np.testing.assert_array_equal(out, values)
        bad = np.array([5.0, np.nan, 7.0], dtype=np.float32)
        assert core._pd32_linear_set_array(handle, 1, 4, bad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 0
        core._pd32_linear_get(handle, 1, 4, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        np.testing.assert_array_equal(out, values)
        assert core._pd32_linear_set_scalar(handle, 2, 4, ctypes.c_float(8.0)) == 1
        assert core._pd32_geti(handle, 3) == 8.0
    finally:
        core._pd32_free(handle)

def test_pdegrid32_rectangular_regions():
    dims = np.array([4, 5, 3], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 3)
    assert handle
    args = (handle, 1, 3, 2, 5, 1, 3)
    try:
        values = np.arange(12, dtype=np.float32) + 1
        out = np.empty_like(values)
        assert core._pd32_region_set_array(*args, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 1
        assert core._pd32_region_get(*args, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 1
        np.testing.assert_array_equal(out, values)
        bad = values.copy()
        bad[7] = np.nan
        assert core._pd32_region_set_array(*args, bad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 0
        assert core._pd32_region_get(*args, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 1
        np.testing.assert_array_equal(out, values)
        assert core._pd32_region_set_scalar(*args, ctypes.c_float(4.5)) == 1
        assert core._pd32_region_get(*args, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 1
        np.testing.assert_array_equal(out, np.full(12, 4.5, dtype=np.float32))
    finally:
        core._pd32_free(handle)
