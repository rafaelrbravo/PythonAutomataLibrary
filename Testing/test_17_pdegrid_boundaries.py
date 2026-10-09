"""PDEGrid multidimensional boundary and anisotropic-spacing validation.

These tests target indexing/orientation mistakes that 1D references cannot
expose. Expected one-step values are computed independently from finite-volume
face fluxes, including PAL's half-cell Dirichlet boundary convention.
"""
import numpy as np
import pytest


def _diffusion_dirichlet_reference(field, rate, dt, spacing, bcs):
    f = np.asarray(field, dtype=np.float64)
    out = f.copy()
    for idx in np.ndindex(f.shape):
        delta = 0.0
        for axis, dx in enumerate(spacing):
            s = rate * dt / (dx * dx)
            for sign in (-1, 1):
                nb = list(idx)
                nb[axis] += sign
                if 0 <= nb[axis] < f.shape[axis]:
                    delta += s * (f[tuple(nb)] - f[idx])
                else:
                    bc = bcs.get((axis, sign))
                    if bc is not None:
                        face_idx = idx[:axis] + idx[axis + 1:]
                        value = float(bc) if np.ndim(bc) == 0 else float(np.asarray(bc)[face_idx])
                        delta += 2.0 * s * (value - f[idx])
        out[idx] += delta
    return out


@pytest.mark.parametrize("shape,spacing", [
    ((4, 5), (0.7, 1.3)),
    ((3, 4, 5), (0.8, 1.1, 1.6)),
])
def test_anisotropic_scalar_diffusion_with_face_bcs_matches_reference(api, shape, spacing):
    rng = np.random.default_rng(1717 + len(shape))
    initial = rng.normal(size=shape).astype(np.float32)
    rate, dt = 0.13, 0.04
    bcs = {
        (0, -1): np.arange(np.prod(shape[1:]), dtype=np.float32).reshape(shape[1:]) / 7 + 1,
        (0, 1): -0.75,
        (1, -1): np.arange(shape[0] * (shape[2] if len(shape) == 3 else 1), dtype=np.float32).reshape((shape[0],) + (() if len(shape) == 2 else (shape[2],))) / 9,
        (1, 1): 0.4,
    }
    kwargs = {"xMinBC": bcs[(0, -1)], "xMaxBC": bcs[(0, 1)], "yMinBC": bcs[(1, -1)], "yMaxBC": bcs[(1, 1)]}
    if len(shape) == 3:
        bcs[(2, -1)] = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape[:2]) / 11 - 0.2
        bcs[(2, 1)] = 1.25
        kwargs.update(zMinBC=bcs[(2, -1)], zMaxBC=bcs[(2, 1)])

    g = api.NewPDEgrid(shape)
    g[tuple(slice(None) for _ in shape)] = initial
    g.SetTimeSpaceStep(dt, *spacing)
    g.Diffusion(rate, **kwargs)
    g.Update()
    expected = _diffusion_dirichlet_reference(initial, rate, dt, spacing, bcs)
    np.testing.assert_allclose(g[tuple(slice(None) for _ in shape)], expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("shape,axis", [((3, 4), 0), ((3, 4), 1), ((3, 4, 5), 0), ((3, 4, 5), 1), ((3, 4, 5), 2)])
def test_boundary_face_array_orientation(api, shape, axis):
    initial = np.zeros(shape, dtype=np.float32)
    face_shape = shape[:axis] + shape[axis + 1:]
    face = (np.arange(np.prod(face_shape), dtype=np.float32).reshape(face_shape) + 1) / 10
    kwargs = [{0: "xMinBC", 1: "yMinBC", 2: "zMinBC"}[axis]]
    g = api.NewPDEgrid(shape)
    g[tuple(slice(None) for _ in shape)] = initial
    g.SetTimeSpaceStep(0.05, 1.0, 1.0, 1.0)
    g.Diffusion(0.2, **{kwargs: face})
    g.Update()
    expected = np.zeros(shape, dtype=np.float64)
    sl = [slice(None)] * len(shape); sl[axis] = 0
    expected[tuple(sl)] = 2 * 0.05 * 0.2 * face
    np.testing.assert_allclose(g[tuple(slice(None) for _ in shape)], expected, rtol=1e-6, atol=1e-7)


def test_safe_rejects_wrong_boundary_face_shape(api, safe_mode):
    if not safe_mode:
        pytest.skip("Fast mode assumes valid boundary arrays")
    g = api.NewPDEgrid((3, 4, 5))
    with pytest.raises(ValueError):
        g.Diffusion(0.1, xMinBC=np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize("method", ["Diffusion", "DiffusionField", "DiffusionInterfaces"])
def test_constant_field_matching_all_dirichlet_boundaries_is_fixed_point(api, method):
    shape = (4, 5, 3)
    value = 2.75
    g = api.NewPDEgrid(shape)
    g[:, :, :] = value
    g.SetTimeSpaceStep(0.03, 0.8, 1.1, 1.4)
    kwargs = dict(xMinBC=value, xMaxBC=value, yMinBC=value, yMaxBC=value, zMinBC=value, zMaxBC=value)
    if method == "Diffusion":
        g.Diffusion(0.12, **kwargs)
    elif method == "DiffusionField":
        g.DiffusionField(np.full(shape, 0.12, dtype=np.float32), **kwargs)
    else:
        rates = np.full(shape, 0.12, dtype=np.float32)
        g.DiffusionInterfaces(rates, rates, rates, **kwargs)
    g.Update()
    np.testing.assert_allclose(g[:, :, :], value, rtol=0, atol=1e-6)
