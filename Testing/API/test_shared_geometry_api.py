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
    if len(dims) > 1:
        assert g.yDim == dims[1]
    if len(dims) > 2:
        assert g.zDim == dims[2]
    for i in range(len(g)):
        coords = [g.ItoX(i)]
        if len(dims) > 1:
            coords.append(g.ItoY(i))
        if len(dims) > 2:
            coords.append(g.ItoZ(i))
        assert g.ToI(*coords) == i


@pytest.mark.parametrize("kind", ["AgentGrid", "PopGrid", "PDEgrid"])
def test_safe_1d_itox_matches_linear_index(api, safe_mode, kind):
    if not safe_mode:
        pytest.skip("Fast mode already has direct 1D ItoX behavior")
    g = getattr(api, "New" + kind)((5,))
    for i in range(5):
        assert g.ItoX(i) == i


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_box_positional_python_contract(api, kind):
    g = _new(api, kind, (5, 6))
    positional = list(g.Box(1, 4, 2, 5))
    assert len(positional) == 9


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
def test_hood_positional_python_contract(api, kind):
    hood = api.VonNeumannHood(2, True)
    g = _new(api, kind, (-5, -6))
    assert len(list(g.Hood(hood, 0, 0))) > 0


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


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
@pytest.mark.parametrize("dims", [(5,), (4, 5), (3, 4, 5)])
def test_python_njit_box_and_hood_coordinate_parity(api, kind, dims):
    """Compare the exact ordered site sequences, including wrapped duplicates."""
    g = _new(api, kind, tuple(-d for d in dims))
    hood = api.MooreHood(len(dims), True)
    center = tuple(0 for _ in dims)
    bounds = tuple(v for d in dims for v in (-1, 2))
    python_box = [g.ToI(*((p,) if len(dims) == 1 else p)) for p in g.Box(*bounds)]
    python_hood = [g.ToI(*((p,) if len(dims) == 1 else p)) for p in g.Hood(hood, *center)]

    if len(dims) == 1:
        @api.njit
        def collect(grid):
            a = []
            for x in grid.Box(-1, 2):
                a.append(grid.ToI(x))
            b = []
            for x in grid.Hood(hood, 0):
                b.append(grid.ToI(x))
            return a, b
    elif len(dims) == 2:
        @api.njit
        def collect(grid):
            a = []
            for x, y in grid.Box(-1, 2, -1, 2):
                a.append(grid.ToI(x, y))
            b = []
            for x, y in grid.Hood(hood, 0, 0):
                b.append(grid.ToI(x, y))
            return a, b
    else:
        @api.njit
        def collect(grid):
            a = []
            for x, y, z in grid.Box(-1, 2, -1, 2, -1, 2):
                a.append(grid.ToI(x, y, z))
            b = []
            for x, y, z in grid.Hood(hood, 0, 0, 0):
                b.append(grid.ToI(x, y, z))
            return a, b
    compiled_box, compiled_hood = collect(g)
    assert list(compiled_box) == python_box
    assert list(compiled_hood) == python_hood


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_python_hood_unroll_keyword_matches_default(api, kind):
    g = _new(api, kind, (4, 5))
    hood = api.VonNeumannHood(2, True)
    expected = list(g.Hood(hood, 1, 2))
    assert expected == [(0, 2), (2, 2), (1, 1), (1, 3)]
    assert list(g.Hood(hood, 1, 2, unroll=True)) == expected


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_python_box_named_bounds_match_positional(api, kind):
    g = _new(api, kind, (4, 5))
    expected = list(g.Box(1, 3, 1, 3))
    assert expected == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert list(g.Box(x1=1, x2=3, y1=1, y2=3)) == expected


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_python_hood_named_coordinates_match_positional(api, kind):
    g = _new(api, kind, (4, 5))
    hood = api.VonNeumannHood(2, True)
    expected = list(g.Hood(hood, 1, 2))
    assert len(expected) == 4
    assert list(g.Hood(hood=hood, x=1, y=2)) == expected


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_compiled_hood_named_coordinate_forms_match_positional(api, kind):
    """Exercise compiled keyword binding that ordinary Python currently rejects."""
    g = _new(api, kind, (4, 5))
    hood = api.VonNeumannHood(2, True)

    @api.njit
    def collect(grid):
        positional = []
        named = []
        for x, y in grid.Hood(hood, 1, 2):
            positional.append(grid.ToI(x, y))
        for x, y in grid.Hood(hood=hood, x=1, y=2):
            named.append(grid.ToI(x, y))
        return positional, named

    positional, named = collect(g)
    assert list(named) == list(positional)
    assert list(named) == [g.ToI(*xy) for xy in g.Hood(hood, 1, 2)]


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_compiled_box_named_bounds_match_positional(api, kind):
    """Verify the compiled side of the pending Box keyword discrepancy."""
    g = _new(api, kind, (4, 5))

    @api.njit
    def collect(grid):
        positional = []
        named = []
        for x, y in grid.Box(1, 3, 1, 3):
            positional.append(grid.ToI(x, y))
        for x, y in grid.Box(x1=1, x2=3, y1=1, y2=3):
            named.append(grid.ToI(x, y))
        return positional, named

    positional, named = collect(g)
    assert list(named) == list(positional)
    assert list(named) == [g.ToI(*xy) for xy in g.Box(1, 3, 1, 3)]


@pytest.mark.parametrize("kind", ["Grid", "PopGrid", "PDEgrid"])
def test_compiled_hood_unroll_matches_default_sites(api, kind):
    """Compare ordered Hood results with and without compiled unroll."""
    g = _new(api, kind, (4, 5))
    hood = api.VonNeumannHood(2, True)

    @api.njit
    def collect(grid):
        ordinary = []
        unrolled = []
        for x, y in grid.Hood(hood, 1, 2):
            ordinary.append(grid.ToI(x, y))
        for x, y in grid.Hood(hood, 1, 2, unroll=True):
            unrolled.append(grid.ToI(x, y))
        return ordinary, unrolled

    ordinary, unrolled = collect(g)
    assert list(unrolled) == list(ordinary)
    assert list(unrolled) == [g.ToI(*xy) for xy in g.Hood(hood, 1, 2)]
