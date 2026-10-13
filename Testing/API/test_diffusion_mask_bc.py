"""Boundary-condition regression tests for masked diffusion."""
import numpy as np
import pytest
import PythonAutomataLibrary as pal


def test_mask_boundary_values_and_impermeable_cells():
    g = pal.NewPDEgrid((3,), wrapX=False)
    g.SetTimeSpaceStep(1.0, 0.1)
    g.DiffusionMask(1.0, np.array([1.0, -1.0, 1.0]), xMinBC=10.0, xMaxBC=20.0)
    g.Update()
    assert g[0] == pytest.approx(2.0)
    assert g[1] == pytest.approx(0.0)
    assert g[2] == pytest.approx(4.0)


def test_mask_default_matches_no_flux():
    g = pal.NewPDEgrid((3,), wrapX=False)
    g.SetTimeSpaceStep(1.0, 0.1)
    g.DiffusionMask(1.0, np.ones(3))
    g.Update()
    assert np.array([g[i] for i in range(3)]) == pytest.approx([0.0, 0.0, 0.0])
