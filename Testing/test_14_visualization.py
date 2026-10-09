"""Visualization API validation.

PixWindow tests use the display-free worker. OpenGL is tested only when a
standalone context is available; lack of EGL/OSMesa is an environment skip,
not a PAL correctness result.
"""
from pathlib import Path

import numpy as np
import pytest


def test_pix_geometry_and_indexing(api):
    pix, win = api.StartPixWindow(5, 4, headless=True)
    try:
        assert len(pix) == 20
        assert pix.Xdim() == 5 and pix.Ydim() == 4
        for x in range(5):
            for y in range(4):
                i = pix.ToI(x, y)
                assert i == x*4 + y
                assert pix.ItoX(i) == x and pix.ItoY(i) == y
    finally:
        win.Close()


def test_pix_headless_save_exact_rgb_orientation(api, tmp_path):
    # PAL colors are packed 0xRRGGBB. Verify both channels and the documented
    # x/y -> image row/column transform by reading the emitted PNG independently.
    pil = pytest.importorskip("PIL.Image")
    pix, win = api.StartPixWindow(3, 2, scale=1, headless=True)
    path = tmp_path / "pix_exact.png"
    try:
        pix[:] = 0
        pix[0, 0] = 0xFF0000
        pix[2, 1] = 0x00FF00
        pix[1, 0] = 0x0000FF
        win.Update()
        win.Save(str(path), block=True)
        arr = np.asarray(pil.open(path).convert("RGB"))
        assert arr.shape == (2, 3, 3)
        # Conversion in PixWindow maps image row yDim-1-y, column x.
        np.testing.assert_array_equal(arr[1, 0], [255, 0, 0])
        np.testing.assert_array_equal(arr[0, 2], [0, 255, 0])
        np.testing.assert_array_equal(arr[1, 1], [0, 0, 255])
    finally:
        win.Close()


def test_pix_async_save_is_flushed_by_close(api, tmp_path):
    pix, win = api.StartPixWindow(4, 3, headless=True)
    path = tmp_path / "async.png"
    pix[:] = 0x123456
    win.Update()
    win.Save(str(path), block=False)
    win.Close()
    assert path.exists() and path.stat().st_size > 0


def test_safe_pix_rejects_out_of_bounds(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast Pix intentionally omits bounds checks")
    pix, win = api.StartPixWindow(2, 3, headless=True)
    try:
        with pytest.raises(IndexError): pix[2, 0] = 0
        with pytest.raises(IndexError): pix[-1] = 0
        with pytest.raises(IndexError): pix[6] = 0
    finally:
        win.Close()


def _start_gl_or_skip(api, *dims):
    try:
        return api.StartOpenGLWindow(*dims, width=96, height=80, headless=True)
    except (RuntimeError, TimeoutError) as exc:
        pytest.skip("standalone OpenGL backend unavailable: " + str(exc))


@pytest.mark.parametrize("dims", [(10, 8), (10, 8, 6)])
def test_opengl_headless_primitives_update_and_save(api, tmp_path, dims):
    pytest.importorskip("moderngl")
    draw, win = _start_gl_or_skip(api, *dims)
    path = tmp_path / ("gl3.png" if len(dims) == 3 else "gl2.png")
    try:
        draw.Background(0x101010)
        draw.Borders(0.15, 0xFFFFFF)
        if len(dims) == 2:
            draw.Circle(1.0, 0xFF0000, 3.0, 4.0)
            draw.BoxSQ(0x00FF00, 5, 2)
            draw.Line(0.2, 0x0000FF, 0.5, 0.5, 8.5, 6.5)
        else:
            draw.Circle(0.8, 0xFF0000, 3.0, 4.0, 2.0)
            draw.BoxSQ(0x00FF00, 5, 2, 3)
            draw.Line(0.2, 0x0000FF, 0.5, 0.5, 8.5, 6.5, 1.0, 4.0)
        win.Update()
        win.Save(str(path), block=True)
        assert path.exists() and path.stat().st_size > 0
    finally:
        win.Close()


def test_safe_visual_constructor_validation(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast visualization assumes valid constructor inputs")
    with pytest.raises(Exception): api.StartPixWindow(0, 5, headless=True)
    with pytest.raises(Exception): api.StartPixWindow(5, 5, scale=0, headless=True)
    with pytest.raises(Exception): api.StartOpenGLWindow(-1, 5, headless=True)
