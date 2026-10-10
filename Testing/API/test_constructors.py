"""Constructor and global-mode API contracts not covered by algorithmic tests.

Run safe and fast in separate fresh processes through Testing/conftest.py.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("constructor,args", [
    ("NewGrid", ((2, 3), np.int8)),
    ("NewAgentGrid", ((2, 3),)),
    ("NewPopGrid", ((2, 3),)),
    ("NewPDEgrid", ((2, 3),)),
])
def test_wrapped_dimensions_report_positive_sizes_and_wrap_flags(api, constructor, args):
    dims = (-2, 3)
    if constructor == "NewGrid":
        obj = getattr(api, constructor)(dims, np.int8)
    else:
        obj = getattr(api, constructor)(dims)
    assert (obj.xDim, obj.yDim, obj.nDims) == (2, 3, 2)
    assert (obj.wrapX, obj.wrapY, obj.wrapZ) == (True, False, False)


@pytest.mark.parametrize("constructor", ["NewGrid", "NewPopGrid", "NewPDEgrid"])
@pytest.mark.parametrize("dims", [(), (0,), (1, 2, 3, 4), (1.5,), (np.nan,), (np.inf,), (2**31,), (-(2**31),)])
def test_spatial_constructor_rejects_invalid_dimensions(api, constructor, dims):
    with pytest.raises(ValueError):
        getattr(api, constructor)(dims, np.int8) if constructor == "NewGrid" else getattr(api, constructor)(dims)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64, np.bool_])
def test_grid_accepts_documented_dtypes(api, dtype):
    g = api.NewGrid((2,), dtype)
    assert len(g) == 2


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, object, "U1"])
def test_grid_rejects_unsupported_dtypes(api, dtype):
    with pytest.raises(ValueError):
        api.NewGrid((2,), dtype)


@pytest.mark.parametrize("props", [-1, 1.5, np.nan, np.inf, 2**31])
def test_agentgrid_rejects_invalid_property_counts(api, props):
    with pytest.raises(ValueError):
        api.NewAgentGrid((2,), numAgentProps=props)


@pytest.mark.parametrize("stackable", [2, -1, 0.5, "yes", None])
def test_agentgrid_rejects_nonboolean_stackable(api, stackable):
    with pytest.raises(ValueError):
        api.NewAgentGrid((2,), isStackable=stackable)


@pytest.mark.parametrize("capacity", [-1, 1.5, np.nan, np.inf, 2**63])
def test_popgrid_rejects_invalid_capacity(api, capacity):
    with pytest.raises(ValueError):
        api.NewPopGrid((2,), capacity=capacity)


def test_agentgrid_allows_zero_dimensional_nonspatial_grid(api):
    g = api.NewAgentGrid(())
    assert g.nDims == 0
    assert len(g) == 0
    assert g.GetPop() == 0


def test_fast_mode_cannot_be_selected_after_constructor_locks_mode(api):
    # In fast-process runs conftest has already selected FastMode, but construction
    # still locks the global choice and a later call must be rejected identically.
    api.NewIList()
    with pytest.raises(RuntimeError):
        api.FastMode()
