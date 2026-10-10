"""Shared Grid/PopGrid/PDEgrid geometry API contracts."""
import numpy as np
import pytest


def _new(api, kind, dims):
    return api.NewGrid(dims, np.int32) if kind == "Grid" else getattr(api, "New" + kind)(dims)


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
@pytest.mark.parametrize("dims", [(5,), (4, 5), (3, 4, 5)])
def test_geometry_metadata_and_coordinate_roundtrip(api, kind, dims):
    g = _new(api, kind, dims)
    assert g.nDims == len(dims)
    assert g.xDim == dims[0]
    assert g.yDim == (dims[1] if len(dims) > 1 else 0)
    assert g.zDim == (dims[2] if len(dims) > 2 else 0)
    for i in range(len(g)):
        coords = [g.ItoX(i)]
        if len(dims) > 1:
            coords.append(g.ItoY(i))
        if len(dims) > 2:
            coords.append(g.ItoZ(i))
        assert g.ToI(*coords) == i


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_box_keyword_and_positional_forms_match(api, kind):
    g = _new(api, kind, (5, 6))
    positional = list(g.Box(1, 4, 2, 5))
    keyword = list(g.Box(x1=1, x2=4, y1=2, y2=5))
    mixed = list(g.Box(1, 4, y1=2, y2=5))
    assert positional == keyword == mixed


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_compiled_box_keyword_and_positional_forms_match(api, kind):
    g = _new(api, kind, (5, 6))

    @api.njit
    def collect(grid):
        a = 0
        b = 0
        for x, y in grid.Box(1, 4, 2, 5):
            a += grid.ToI(x, y) + 1
        for x, y in grid.Box(x1=1, x2=4, y1=2, y2=5):
            b += grid.ToI(x, y) + 1
        return a, b

    a, b = collect(g)
    assert a == b


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_hood_keyword_center_matches_positional(api, kind):
    hood = api.VonNeumannHood(2, True)
    g = _new(api, kind, (-5, -6))
    assert list(g.Hood(hood, 0, 0)) == list(g.Hood(hood, x=0, y=0))


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_safe_invalid_geometry_coordinates(api, safe_mode, kind):
    if not safe_mode:
        pytest.skip("Fast mode intentionally omits geometry validation")
    g = _new(api, kind, (4, 5))
    for call in (
        lambda: g.ToI(-1, 0),
        lambda: g.ToI(4, 0),
        lambda: g.ToI(0, 5),
        lambda: g.ItoX(-1),
        lambda: g.ItoX(len(g)),
    ):
        with pytest.raises((IndexError, ValueError)):
            call()
