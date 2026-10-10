import numpy as np
import pytest

import PythonAutomataLibrary as pal


@pytest.mark.parametrize("method", ["Advection", "AdvectionField", "AdvectionInterfaces"])
def test_advection_wrapped_1d_conserves_mass(method):
    field = pal.NewPDEgrid((-12,))
    field.SetTimeSpaceStep(0.1, 1.0)
    values = np.arange(12, dtype=np.float32) % 5
    field[:] = values
    before = float(np.sum(values, dtype=np.float64))

    if method == "Advection":
        field.Advection(0.4)
    else:
        getattr(field, method)(np.full(12, 0.4, dtype=np.float32))
    field.Update()

    assert float(np.sum(field[:], dtype=np.float64)) == pytest.approx(before, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("method", ["Advection", "AdvectionField", "AdvectionInterfaces"])
def test_advection_wrapped_1d_preserves_constant_field(method):
    field = pal.NewPDEgrid((-12,))
    field.SetTimeSpaceStep(0.1, 1.0)
    field[:] = 2.75

    if method == "Advection":
        field.Advection(-0.4)
    else:
        getattr(field, method)(np.full(12, -0.4, dtype=np.float32))
    field.Update()

    np.testing.assert_allclose(field[:], 2.75, rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", ["Advection", "AdvectionField", "AdvectionInterfaces"])
@pytest.mark.parametrize(("velocity", "low_bc", "high_bc", "edge"), [
    (0.5, 4.0, 9.0, 0),
    (-0.5, 9.0, 4.0, -1),
])
def test_advection_uses_upwind_boundary_as_inflow(method, velocity, low_bc, high_bc, edge):
    field = pal.NewPDEgrid((6,))
    field.SetTimeSpaceStep(0.2, 1.0)
    field[:] = 0.0

    kwargs = {"xMinBC": low_bc, "xMaxBC": high_bc}
    if method == "Advection":
        field.Advection(velocity, **kwargs)
    else:
        getattr(field, method)(np.full(6, velocity, dtype=np.float32), **kwargs)
    field.Update()

    expected = np.zeros(6, dtype=np.float32)
    expected[edge] = 0.4
    np.testing.assert_allclose(field[:], expected, rtol=0, atol=2e-6)
