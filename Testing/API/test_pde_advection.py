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


def test_advection_field_and_interfaces_use_distinct_velocity_locations():
    centered = pal.NewPDEgrid((-4,))
    interfaces = pal.NewPDEgrid((-4,))
    values = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    velocities = np.array([0.2, 0.6, 0.2, 0.2], dtype=np.float32)
    for field in (centered, interfaces):
        field.SetTimeSpaceStep(0.1, 1.0)
        field[:] = values

    centered.AdvectionField(velocities)
    interfaces.AdvectionInterfaces(velocities)
    centered.Update()
    interfaces.Update()

    expected_centered = np.array([0.96, 0.04, 0.0, 0.0], dtype=np.float32)
    expected_interfaces = np.array([0.98, 0.02, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(centered[:], expected_centered, rtol=0, atol=2e-6)
    np.testing.assert_allclose(interfaces[:], expected_interfaces, rtol=0, atol=2e-6)


@pytest.mark.parametrize("method", ["Advection", "AdvectionField", "AdvectionInterfaces"])
def test_advection_cfl_validation_is_transactional(method, safe_mode):
    if not safe_mode:
        pytest.skip("CFL validation is a safe-mode contract")
    field = pal.NewPDEgrid((-8,))
    field.SetTimeSpaceStep(1.0, 1.0)
    values = np.arange(8, dtype=np.float32)
    field[:] = values

    with pytest.raises(ValueError):
        if method == "Advection":
            field.Advection(1.5)
        else:
            getattr(field, method)(np.full(8, 1.5, dtype=np.float32))
    field.Update()

    np.testing.assert_array_equal(field[:], values)
