"""ColorScale public interpolation and compiled-parity contracts."""
import numpy as np
import pytest


def test_colorscale_empty_singleton_and_clamping(api):
    assert int(api.ColorScale((), 0.5)) == 0
    assert int(api.ColorScale((0x123456,), -100.0)) == 0x123456
    assert int(api.ColorScale((0x123456,), 100.0)) == 0x123456
    colors = (0x102030, 0xA0B0C0)
    assert int(api.ColorScale(colors, -1.0)) == colors[0]
    assert int(api.ColorScale(colors, 2.0)) == colors[-1]


@pytest.mark.parametrize("value,expected", [
    (0.0, 0x000000),
    (0.25, 0x404040),
    (0.5, 0x808080),
    (0.75, 0xBFBFBF),
    (1.0, 0xFFFFFF),
])
def test_colorscale_rgb_interpolation_rounding(api, value, expected):
    assert int(api.ColorScale((0x000000, 0xFFFFFF), value)) == expected


def test_colorscale_multistop_hits_exact_stops(api):
    colors = (0xFF0000, 0x00FF00, 0x0000FF)
    assert int(api.ColorScale(colors, 0.0)) == colors[0]
    assert int(api.ColorScale(colors, 0.5)) == colors[1]
    assert int(api.ColorScale(colors, 1.0)) == colors[2]


def test_colorscale_compiled_matches_python(api):
    colors = (0x102030, 0x80A0C0, 0xF0E0D0)

    @api.njit
    def compiled(value):
        return api.ColorScale(colors, value)

    for value in (-0.2, 0.0, 0.1, 0.5, 0.9, 1.0, 1.2):
        assert int(compiled(value)) == int(api.ColorScale(colors, value))
