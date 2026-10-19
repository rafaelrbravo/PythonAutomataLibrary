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
