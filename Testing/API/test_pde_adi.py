import numpy as np
import pytest

import PythonAutomataLibrary as pal


@pytest.mark.parametrize("dimensions", [(8,), (5, 6), (4, 5, 6)])
def test_diffusion_adi_preserves_constant_field(dimensions):
    field = pal.NewPDEgrid(dimensions)
    field.SetTimeSpaceStep(0.3, 1.0, 1.0, 1.0)
    field[:] = 2.75

    field.DiffusionADI(0.4)
    field.Update()

    np.testing.assert_allclose(field[:], 2.75, rtol=0, atol=3e-6)


@pytest.mark.parametrize("dimensions", [(-8,), (-5, -6), (-4, -5, -6)])
def test_diffusion_adi_wrapped_domain_conserves_mass(dimensions):
    field = pal.NewPDEgrid(dimensions)
    field.SetTimeSpaceStep(0.3, 1.0, 1.0, 1.0)
    values = np.arange(len(field), dtype=np.float32) % 7
    field[:] = values
    before = float(np.sum(values, dtype=np.float64))

    field.DiffusionADI(0.4)
    field.Update()

    assert float(np.sum(field[:], dtype=np.float64)) == pytest.approx(before, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize(("dimensions", "boundaries"), [
    ((8,), {"xMinBC": 2.75, "xMaxBC": 2.75}),
    ((5, 6), {"xMinBC": 2.75, "xMaxBC": 2.75, "yMinBC": 2.75, "yMaxBC": 2.75}),
    ((4, 5, 6), {
        "xMinBC": 2.75, "xMaxBC": 2.75,
        "yMinBC": 2.75, "yMaxBC": 2.75,
        "zMinBC": 2.75, "zMaxBC": 2.75,
    }),
])
def test_diffusion_adi_matching_fixed_boundaries_preserve_equilibrium(dimensions, boundaries):
    field = pal.NewPDEgrid(dimensions)
    field.SetTimeSpaceStep(0.3, 1.0, 1.0, 1.0)
    field[:] = 2.75

    field.DiffusionADI(0.4, **boundaries)
    field.Update()

    np.testing.assert_allclose(field[:], 2.75, rtol=0, atol=3e-6)
