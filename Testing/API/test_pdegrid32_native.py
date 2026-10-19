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
        i = core._pd32_toi(handle, 1, 2, 0)
        assert core._pd32_seti(handle, i, np.float32(1.25)) == 1
        core._pd32_addi(handle, i, np.float32(0.5))
        core._pd32_update(handle)
        return core._pd32_geti(handle, i)

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

def test_pdegrid32_conservative_upwind_advection():
    dims = np.array([-5], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
    assert handle
    try:
        assert core._pd32_seti(handle, 2, ctypes.c_float(1.0)) == 1
        assert core._pd32_advection(handle, 0.25, 0.0, 0.0) == 1
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 2) == pytest.approx(0.75)
        assert core._pd32_geti(handle, 3) == pytest.approx(0.25)
        assert core._pd32_advection(handle, 1.1, 0.0, 0.0) == 0
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 2) == pytest.approx(0.75)
        assert core._pd32_geti(handle, 3) == pytest.approx(0.25)
    finally:
        core._pd32_free(handle)

def test_pdegrid32_masked_diffusion_blocks_flux_and_rejects_invalid_mask():
    dims = np.array([5], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
    assert handle
    try:
        assert core._pd32_seti(handle, 2, ctypes.c_float(1.0)) == 1
        mask = np.ones(5, dtype=np.float32)
        mask[3] = -1
        assert core._pd32_diffuse_mask_addr(handle, 0.1, mask.ctypes.data) == 1
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 1) == pytest.approx(0.1, abs=1e-6)
        assert core._pd32_geti(handle, 2) == pytest.approx(0.9, abs=1e-6)
        assert core._pd32_geti(handle, 3) == 0
        mask[4] = np.nan
        assert core._pd32_diffuse_mask_addr(handle, 0.1, mask.ctypes.data) == 0
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 2) == pytest.approx(0.9, abs=1e-6)
    finally:
        core._pd32_free(handle)

def test_pdegrid32_mask_uses_sign_not_magnitude():
    dims = np.array([4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
    assert handle
    try:
        assert core._pd32_seti(handle, 1, ctypes.c_float(1.0)) == 1
        mask = np.array([0.0, 3.0, -1.0, 0.0], dtype=np.float32)
        assert core._pd32_diffuse_mask_addr(handle, 0.1, mask.ctypes.data) == 1
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 0) == pytest.approx(0.1, abs=1e-6)
        assert core._pd32_geti(handle, 1) == pytest.approx(0.9, abs=1e-6)
        assert core._pd32_geti(handle, 2) == 0.0
    finally:
        core._pd32_free(handle)

def test_pdegrid32_variable_rate_harmonic_diffusion():
    dims = np.array([4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
    assert handle
    try:
        assert core._pd32_seti(handle, 1, ctypes.c_float(1.0)) == 1
        rates = np.array([0.0, 0.2, 0.1, 0.0], dtype=np.float32)
        ptr = rates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        assert core._pd32_diffuse_field_addr(handle, rates.ctypes.data) == 1
        core._pd32_update(handle)
        expected = 2 * 0.2 * 0.1 / (0.2 + 0.1)
        assert core._pd32_geti(handle, 1) == pytest.approx(1 - expected, abs=1e-6)
        assert core._pd32_geti(handle, 2) == pytest.approx(expected, abs=1e-6)
        rates[3] = np.nan
        assert core._pd32_diffuse_field_addr(handle, rates.ctypes.data) == 0
        core._pd32_update(handle)
        assert core._pd32_geti(handle, 2) == pytest.approx(expected, abs=1e-6)
    finally:
        core._pd32_free(handle)

def test_pdegrid32_jitclass_handles():
    dims = np.array([5], dtype=np.int32)
    for cls in (core._PDEgrid32Safe, core._PDEgrid32Fast):
        handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
        assert handle
        try:
            grid = cls(handle, 1, 5, -1, -1, 5, False, False, False)
            grid.SetI(2, np.float32(1.0))
            grid.AddI(np.float32(0.25), 2)
            grid.Update()
            assert grid.GetI(2) == pytest.approx(1.25)
            @core.nb.njit
            def read_compiled(g):
                return g.GetI(2)
            assert read_compiled(grid) == pytest.approx(1.25)
        finally:
            core._pd32_free(handle)

def test_pdegrid32_jitclass_integer_indexing():
    dims = np.array([4], dtype=np.int32)
    for cls in (core._PDEgrid32Safe, core._PDEgrid32Fast):
        handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
        assert handle
        try:
            grid = cls(handle, 1, 4, -1, -1, 4, False, False, False)
            grid[1] = np.float32(2.5)
            assert grid[1] == pytest.approx(2.5)
            @core.nb.njit
            def compiled_index(g):
                g[2] = np.float32(3.25)
                return g[2]
            assert compiled_index(grid) == pytest.approx(3.25)
        finally:
            core._pd32_free(handle)

def test_pdegrid32_coordinate_tuple_indexing():
    dims = np.array([3, 4], dtype=np.int32)
    for cls in (core._PDEgrid32Safe, core._PDEgrid32Fast):
        handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2)
        assert handle
        try:
            grid = cls(handle, 2, 3, 4, -1, 12, False, False, False)
            grid[1, 2] = np.float32(4.5)
            assert grid[1, 2] == pytest.approx(4.5)
            @core.nb.njit
            def compiled_coordinates(g):
                g[2, 3] = np.float32(1.25)
                return g[2, 3]
            assert compiled_coordinates(grid) == pytest.approx(1.25)
        finally:
            core._pd32_free(handle)

def test_pdegrid32_safe_coordinate_validation():
    dims = np.array([3, 4], dtype=np.int32)
    handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2)
    assert handle
    try:
        grid = core._PDEgrid32Safe(handle, 2, 3, 4, -1, 12, False, False, False)
        with pytest.raises(IndexError):
            grid[3, 0] = np.float32(1.0)
        with pytest.raises(IndexError):
            grid[-1, 0] = np.float32(1.0)
        with pytest.raises(ValueError):
            grid[1.5, 0] = np.float32(1.0)
    finally:
        core._pd32_free(handle)

def test_internal_pdegrid32_factory_respects_geometry_and_dtype():
    for dims in ((5,), (3, 4), (2, 3, 4), (-5,)):
        grid = core._NewPDEgrid32(dims)
        try:
            assert len(grid) == int(np.prod(np.abs(dims)))
            assert grid.nDims == len(dims)
            assert grid.wrapX == (dims[0] < 0)
            grid[0] = np.float32(1.5)
            assert grid[0] == pytest.approx(1.5)
        finally:
            core._pd32_free(grid._ptr)

def test_pdegrid32_jitclass_mask_and_field_diffusion():
    for cls in (core._PDEgrid32Safe, core._PDEgrid32Fast):
        dims = np.array([5], dtype=np.int32)
        handle = core._pd32_new(dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1)
        assert handle
        try:
            grid = cls(handle, 1, 5, -1, -1, 5, False, False, False)
            grid[2] = np.float32(1.0)
            grid.DiffusionMask(0.1, np.array([0, 0, 0, -1, 0], dtype=np.float32))
            grid.Update()
            assert grid[1] == pytest.approx(0.1, abs=1e-6)
            assert grid[3] == 0.0
            grid.Reset()
            grid[2] = np.float32(1.0)
            grid.DiffusionField(np.array([0, 0.1, 0.2, 0.0, 0], dtype=np.float32))
            grid.Update()
            assert grid[1] == pytest.approx(2 * 0.1 * 0.2 / 0.3, abs=1e-6)
            @core.nb.njit
            def compiled_mask(g, m):
                g.Reset()
                g[2] = np.float32(1.0)
                g.DiffusionMask(0.1, m)
                g.Update()
                return g[1]
            assert compiled_mask(grid, np.array([0, 0, 0, -1, 0], dtype=np.float32)) == pytest.approx(0.1, abs=1e-6)
        finally:
            core._pd32_free(handle)


@pytest.mark.parametrize("dimensions", [(7,), (-7,), (4, 5), (-4, 5), (3, -4, 5), (1, 5)])
def test_shared_diffusion_precision_parity_and_mass_conservation(dimensions):
    """The shared face-flux kernel must agree across storage precisions."""
    ptr, i32 = ctypes.c_uint64, ctypes.c_int32
    size = int(np.prod(np.abs(dimensions)))
    results = []
    for prefix, scalar, dtype, method in (
        ("pal_pd32_", ctypes.c_float, np.float32, "diffuse"),
        ("pal_pd_", ctypes.c_double, np.float64, "diffusion"),
    ):
        new = _native(prefix + "new", ptr, ctypes.POINTER(i32), i32)
        free = _native(prefix + "free", None, ptr)
        seti = _native(prefix + "seti", i32, ptr, i32, scalar)
        diffuse = _native(prefix + method, i32, ptr, ctypes.c_double)
        update = _native(prefix + "update", None, ptr)
        geti = _native(prefix + "geti", scalar, ptr, i32)
        handle = new((i32 * len(dimensions))(*dimensions), len(dimensions))
        assert handle
        try:
            
            initial = np.linspace(0.25, 1.25, size, dtype=dtype)
            for i, value in enumerate(initial):
                assert seti(handle, i, scalar(value)) == 1
            mass_before = initial.sum(dtype=np.float64)
            assert diffuse(handle, 0.05) == 1
            update(handle)
            values = np.array([geti(handle, i) for i in range(size)], dtype=dtype)
            assert values.sum(dtype=np.float64) == pytest.approx(mass_before, abs=2e-5)
            results.append(values)
        finally:
            free(handle)
    np.testing.assert_allclose(results[0], results[1], atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("prefix,scalar,method", [
    ("pal_pd32_", ctypes.c_float, "diffuse"),
    ("pal_pd_", ctypes.c_double, "diffusion"),
])
def test_shared_diffusion_periodic_edge_transfer(prefix, scalar, method):
    ptr, i32 = ctypes.c_uint64, ctypes.c_int32
    new = _native(prefix + "new", ptr, ctypes.POINTER(i32), i32)
    free = _native(prefix + "free", None, ptr)
    seti = _native(prefix + "seti", i32, ptr, i32, scalar)
    diffuse = _native(prefix + method, i32, ptr, ctypes.c_double)
    update = _native(prefix + "update", None, ptr)
    get = _native(prefix + "geti", scalar, ptr, i32)
    for signed_dim, expected_last in ((-5, 0.1), (5, 0.0)):
        handle = new((i32 * 1)(signed_dim), 1)
        assert handle
        try:
            assert seti(handle, 0, scalar(1.0)) == 1
            assert diffuse(handle, 0.1) == 1
            update(handle)
            assert get(handle, 4) == pytest.approx(expected_last, abs=1e-6)
            assert get(handle, 1) == pytest.approx(0.1, abs=1e-6)
        finally:
            free(handle)


@pytest.mark.parametrize("prefix,scalar,method", [
    ("pal_pd32_", ctypes.c_float, "diffuse"),
    ("pal_pd_", ctypes.c_double, "diffusion"),
])
def test_shared_diffusion_invalid_rate_is_atomic(prefix, scalar, method):
    ptr, i32 = ctypes.c_uint64, ctypes.c_int32
    new = _native(prefix + "new", ptr, ctypes.POINTER(i32), i32)
    free = _native(prefix + "free", None, ptr)
    seti = _native(prefix + "seti", i32, ptr, i32, scalar)
    diffuse = _native(prefix + method, i32, ptr, ctypes.c_double)
    update = _native(prefix + "update", None, ptr)
    geti = _native(prefix + "geti", scalar, ptr, i32)
    handle = new((i32 * 2)(5, 5), 2)
    assert handle
    try:
        assert seti(handle, 12, scalar(1.0)) == 1
        for rate in (-0.1, float("nan"), float("inf"), 0.3):
            assert diffuse(handle, rate) == 0
        update(handle)
        assert geti(handle, 12) == pytest.approx(1.0)
        assert sum(geti(handle, i) for i in range(25)) == pytest.approx(1.0)
    finally:
        free(handle)


@pytest.mark.parametrize("safe", [False, True])
def test_internal_float32_pdegrid_jitclass_in_compiled_model(safe):
    """Exercise the actual float32 PDEgrid object through a compiled model."""
    previous = core._UseFastMode
    core._UseFastMode = lambda: not safe
    try:
        grid = core._NewPDEgrid32((5,))
        @core.nb.njit
        def model(field):
            field.SetTimeSpaceStep(1.0, 1.0)
            field.SetI(2, np.float32(1.0))
            field.Diffusion(0.1)
            field.Update()
            return field.GetI(1), field.GetI(2), field.GetI(3)
        left, center, right = model(grid)
        assert (left, center, right) == pytest.approx((0.1, 0.8, 0.1), abs=1e-6)
    finally:
        core._UseFastMode = previous
        if "grid" in locals():
            core._pd32_free(grid._ptr)


def test_pdegrid32_add_coordinates_in_compiled_jitclass():
    from NativeCore import _NewPDEgrid32

    @core.nb.njit
    def run(g):
        g.Add(0.75, 1, 2)
        g.Update()
        return g[1, 2]

    original = core._UseFastMode
    try:
        for fast in (False, True):
            core._UseFastMode = lambda: fast
            grid = _NewPDEgrid32([3, 4])
            try:
                assert run(grid) == pytest.approx(0.75, abs=1e-6)
            finally:
                core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original


def test_pdegrid32_safe_steps_reject_invalid_values():
    from NativeCore import _NewPDEgrid32

    original = core._UseFastMode
    core._UseFastMode = lambda: False
    try:
        grid = _NewPDEgrid32([4])
        try:
            for invalid in (0.0, -1.0, np.nan, np.inf):
                with pytest.raises(ValueError):
                    grid.SetTimeSpaceStep(1.0, invalid)
                with pytest.raises(ValueError):
                    grid.SetTimeSpaceStep(invalid, 1.0)
            grid.SetTimeSpaceStep(0.5, 2.0)
            grid[1] = 1.0
            grid.Diffusion(0.1)
            grid.Update()
            assert grid[0] == pytest.approx(0.0125, abs=1e-6)
        finally:
            core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original


def test_pdegrid32_step_accessors_compiled_in_both_modes():
    from NativeCore import _NewPDEgrid32

    @core.nb.njit
    def read_steps(grid):
        return grid.Dt(), grid.Dx(), grid.Dy(), grid.Dz()

    original = core._UseFastMode
    try:
        for fast in (False, True):
            core._UseFastMode = lambda: fast
            grid = _NewPDEgrid32([2, 3, 4])
            try:
                grid.SetTimeSpaceStep(0.5, 1.5, 2.5, 3.5)
                assert read_steps(grid) == pytest.approx((0.5, 1.5, 2.5, 3.5))
            finally:
                core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original


def test_pdegrid32_wrap_helpers_match_periodic_and_fixed_boundaries():
    from NativeCore import _NewPDEgrid32

    @core.nb.njit
    def run(g):
        return g.InWrapX(-1), g.InWrapX(4), g.InWrapY(-1), g.InWrapY(3)

    original = core._UseFastMode
    try:
        for fast in (False, True):
            core._UseFastMode = lambda: fast
            for dims, expected in [([4, 3], (-2147483648, -2147483648, -2147483648, -2147483648)), ([-4, -3], (3, 0, 2, 0))]:
                grid = _NewPDEgrid32(dims)
                try:
                    assert run(grid) == expected
                finally:
                    core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original


def test_pdegrid32_safe_reverse_coordinates_reject_invalid_indices():
    from NativeCore import _NewPDEgrid32

    original = core._UseFastMode
    core._UseFastMode = lambda: False
    try:
        grid = _NewPDEgrid32([3])
        try:
            assert grid.ItoX(2) == 2
            for invalid in (-1, 3, np.nan, np.inf, 0.5):
                with pytest.raises((ValueError, IndexError)):
                    grid.ItoX(invalid)
            with pytest.raises(ValueError):
                grid.ItoY(0)
            with pytest.raises(ValueError):
                grid.ItoZ(0)
        finally:
            core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original


def test_pdegrid32_safe_linear_access_validates_indices_and_values():
    original = core._UseFastMode
    core._UseFastMode = lambda: False
    try:
        grid = core._NewPDEgrid32([3])
        try:
            assert grid.SetI(1, 1.25)
            assert grid.GetI(1) == pytest.approx(1.25)
            grid.AddI(0.5, 1)
            grid.Update()
            assert grid.GetI(1) == pytest.approx(1.75)
            for invalid in (-1, 3, np.nan, np.inf, 0.5):
                with pytest.raises((ValueError, IndexError)):
                    grid.GetI(invalid)
                with pytest.raises((ValueError, IndexError)):
                    grid.SetI(invalid, 2)
                with pytest.raises((ValueError, IndexError)):
                    grid.AddI(2, invalid)
            for invalid in (np.nan, np.inf):
                with pytest.raises(ValueError):
                    grid.SetI(1, invalid)
                with pytest.raises(ValueError):
                    grid.AddI(invalid, 1)
            assert grid.GetI(1) == pytest.approx(1.75)
        finally:
            core._pd32_free(grid._ptr)
    finally:
        core._UseFastMode = original
