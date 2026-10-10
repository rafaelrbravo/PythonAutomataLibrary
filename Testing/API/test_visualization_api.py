"""Visualization API contracts that are safe to exercise headlessly."""
import numpy as np
import pytest


def test_pix_compiled_write_geometry_and_slices(api, tmp_path):
    pix, win = api.StartPixWindow(5, 4, headless=True)
    try:
        @api.njit
        def draw(p):
            p[:] = 0
            p[1:4] = 0x010203
            p[2, 3] = 0xA0B0C0
            return len(p), p.Xdim(), p.Ydim(), p.ToI(2, 3), p.ItoX(11), p.ItoY(11)

        assert draw(pix) == (20, 5, 4, 11, 2, 3)
        win.Update()
        out = tmp_path / "compiled.png"
        win.Save(str(out), block=True)
        image = pytest.importorskip("PIL.Image").open(out).convert("RGB")
        arr = np.asarray(image)
        np.testing.assert_array_equal(arr[0, 2], [0xA0, 0xB0, 0xC0])
    finally:
        win.Close()


def test_pix_headless_gif_lifecycle(api, tmp_path):
    pytest.importorskip("PIL.Image")
    pix, win = api.StartPixWindow(3, 2, headless=True)
    path = tmp_path / "anim.gif"
    try:
        assert win.StartGif(str(path), delay=25) is win
        pix[:] = 0xFF0000
        win.Update()
        assert win.AddGifFrame() is win
        pix[:] = 0x00FF00
        win.Update()
        win.AddGifFrame(block=True)
        assert win.StopGif() is win
        assert path.exists() and path.stat().st_size > 0
    finally:
        win.Close()


def test_safe_pixwindow_lifecycle_validation(api, safe_mode, tmp_path):
    if not safe_mode:
        pytest.skip("Fast visualization intentionally omits lifecycle validation")
    pix, win = api.StartPixWindow(2, 2, headless=True)
    path = tmp_path / "x.gif"
    try:
        with pytest.raises(Exception): win.AddGifFrame()
        with pytest.raises(Exception): win.StopGif()
        win.StartGif(str(path))
        with pytest.raises(Exception): win.StartGif(str(path))
        with pytest.raises(Exception): win.Save(123)
        with pytest.raises(Exception): win.Save(str(tmp_path / "x.png"), block="yes")
    finally:
        win.Close()
    with pytest.raises(Exception): win.Close()


@pytest.mark.parametrize("kwargs", [
    dict(xDim=2.5, yDim=3, headless=True),
    dict(xDim=2, yDim=3.5, headless=True),
    dict(xDim=2, yDim=3, scale=1.5, headless=True),
    dict(xDim=2, yDim=3, title=5, headless=True),
    dict(xDim=2, yDim=3, headless=1),
])
def test_safe_pix_constructor_type_validation(api, safe_mode, kwargs):
    if not safe_mode:
        pytest.skip("Fast visualization assumes valid constructor inputs")
    with pytest.raises(Exception):
        api.StartPixWindow(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(xDim=2, yDim=3, width=2.5, height=10, headless=True),
    dict(xDim=2, yDim=3, width=10, height=10.5, headless=True),
    dict(xDim=2, yDim=3, title=5, headless=True),
    dict(xDim=2, yDim=3, headless=1),
])
def test_safe_opengl_constructor_validation_without_context(api, safe_mode, kwargs):
    if not safe_mode:
        pytest.skip("Fast visualization assumes valid constructor inputs")
    with pytest.raises(Exception):
        api.StartOpenGLWindow(**kwargs)


def test_awaitwindows_headless_multiple_window_lifecycle(api):
    pix_a, win_a = api.StartPixWindow(2, 2, headless=True)
    pix_b, win_b = api.StartPixWindow(3, 2, headless=True)
    try:
        assert api.AwaitWindows(timeout=5) is None
        assert win_a.IsOpen()
        assert win_b.IsOpen()
        pix_a[:] = 0x112233
        pix_b[:] = 0x445566
        win_a.Update()
        win_b.Update()
    finally:
        win_a.Close()
        win_b.Close()
    assert api.AwaitWindows(timeout=0.1) is None


def _start_headless_opengl_or_skip(api, *args, **kwargs):
    pytest.importorskip("moderngl")
    try:
        return api.StartOpenGLWindow(*args, headless=True, **kwargs)
    except RuntimeError as exc:
        if "could not create a headless OpenGL context" in str(exc):
            pytest.skip(str(exc))
        raise


def test_opengl_headless_2d_render_save(api, tmp_path):
    pytest.importorskip("PIL.Image")
    draw, win = _start_headless_opengl_or_skip(api, 4, 3, width=64, height=48)
    out = tmp_path / "opengl2d.png"
    try:
        draw.BoxSQ(0xFF0000, 1, 1)
        draw.Circle(0.4, 0x00FF00, 2.5, 1.5)
        win.Update()
        assert win.Save(str(out), block=True) is win
        image = pytest.importorskip("PIL.Image").open(out).convert("RGB")
        assert image.size == (64, 48)
        assert np.asarray(image).max() > 0
    finally:
        win.Close()


def test_opengl_headless_3d_render_save(api, tmp_path):
    pytest.importorskip("PIL.Image")
    draw, win = _start_headless_opengl_or_skip(api, 4, 3, 2, width=64, height=48)
    out = tmp_path / "opengl3d.png"
    try:
        draw.Box(1.0, 0x3366FF, 2.0, 1.5, 1.0)
        draw.Line(0.1, 0xFFFFFF, 0.0, 0.0, 4.0, 3.0, 0.0, 2.0)
        win.Update()
        assert win.Save(str(out), block=True) is win
        image = pytest.importorskip("PIL.Image").open(out).convert("RGB")
        assert image.size == (64, 48)
        assert np.asarray(image).max() > 0
    finally:
        win.Close()
