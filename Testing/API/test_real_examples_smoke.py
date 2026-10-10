"""Headless smoke tests for representative real PAL examples."""

import importlib.util
from pathlib import Path

import numpy as np
import PythonAutomataLibrary as pal

ROOT = Path(__file__).parents[2]


def _load(relative):
    path = ROOT / relative
    name = "pal_example_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_birthdeath_real_example_kernels():
    ns = _load("Examples/Agents/BirthDeath.py")
    grid = pal.NewAgentGrid((ns.X, ns.Y), numAgentProps=0, isStackable=False)
    empty = pal.NewIList()
    ns.Setup(grid)
    before = grid.GetPop()
    assert before > 0
    pal.Seed(1)
    ns.Step(grid, empty)
    after = grid.GetPop()
    assert 0 <= after <= grid.xDim * grid.yDim


def test_popgrid_multinomial_real_example_kernel():
    ns = _load("Examples/Agents/PopGridExample.py")
    cells = pal.NewPopGrid((ns.X, ns.Y))
    mn = pal.NewMultinomial()
    cells[0, 0] = ns.STARTING_POP
    before = cells.GetPop()
    pal.Seed(2)
    ns.Step(cells, mn)
    cells.Update()
    assert cells.GetPop() == before
    assert all(cells[i] >= 0 for i in range(len(cells)))


def test_reaction_diffusion_2d_real_example_kernel():
    ns = _load("Examples/Diffusibles/ReactionDiffusion2D.py")
    g1 = pal.NewPDEgrid((ns.X, ns.Y))
    g2 = pal.NewPDEgrid((ns.X, ns.Y))
    ns.Step(g1, g2)
    a = np.array([g1[i] for i in range(len(g1))])
    b = np.array([g2[i] for i in range(len(g2))])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.max() > 0 and b.max() > 0


def test_diffusion_advection_3d_real_example_kernel():
    ns = _load("Examples/Diffusibles/DiffusionAdvection3D.py")
    grid = pal.NewPDEgrid((-ns.X, -ns.Y, -ns.Z))
    grid[ns.X // 2, ns.Y // 2, ns.Z // 2] = 1.0
    ns.Step(grid)
    values = np.array([grid[i] for i in range(len(grid))])
    assert np.isfinite(values).all()
    assert values.min() >= 0
    assert values.sum() > 0
