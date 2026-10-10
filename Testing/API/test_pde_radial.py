import numpy as np
import pytest

import PythonAutomataLibrary as pal


def _radial_weights(n, dim):
    edges = np.arange(n + 1, dtype=np.float64)
    return edges[1:] ** dim - edges[:-1] ** dim


@pytest.mark.parametrize(("method", "dim"), [
    ("DiffusionRadialCircle", 2),
    ("DiffusionRadialSphere", 3),
])
def test_radial_diffusion_conserves_weighted_mass_without_outer_bc(method, dim):
    field = pal.NewPDEgrid((8,))
    field.SetTimeSpaceStep(0.1, 1.0)
    values = np.array([1.0, 4.0, 0.5, 2.0, 7.0, 3.0, 0.0, 5.0], dtype=np.float32)
    field[:] = values
    weights = _radial_weights(len(field), dim)
    before = np.dot(values.astype(np.float64), weights)

    getattr(field, method)(0.2)
    field.Update()

    after = np.dot(np.asarray(field[:], dtype=np.float64), weights)
    assert after == pytest.approx(before, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
def test_radial_diffusion_preserves_constant_field(method):
    field = pal.NewPDEgrid((8,))
    field.SetTimeSpaceStep(0.1, 1.0)
    field[:] = 3.25

    getattr(field, method)(0.2)
    field.Update()

    np.testing.assert_allclose(field[:], 3.25, rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", ["DiffusionRadialCircle", "DiffusionRadialSphere"])
def test_radial_diffusion_matching_outer_bc_preserves_constant_field(method):
    field = pal.NewPDEgrid((8,))
    field.SetTimeSpaceStep(0.1, 1.0)
    field[:] = 3.25

    getattr(field, method)(0.2, outerBC=3.25)
    field.Update()

    np.testing.assert_allclose(field[:], 3.25, rtol=0, atol=2e-6)
