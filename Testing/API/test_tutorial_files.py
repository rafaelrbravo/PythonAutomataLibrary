"""Exercise the actual numbered tutorial files without opening desktop windows."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import PythonAutomataLibrary as pal

ROOT = Path(__file__).parents[2] / "Examples" / "Tutorial"
NAMES = ("01_PlaceCells", "02_DrawCells", "03_FindNeighbors", "04_MoveCells", "05_CompileStep", "06_DiffusingNutrient", "07_PopulationCounts")


def load_stage(name):
    path = ROOT / (name + ".py")
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("tutorial_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_tutorial_stage(name, monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_CACHE", "1")
    stage = load_stage(name)
    if name == "01_PlaceCells":
        assert len(stage.Setup().All()) == 4
        return

    if name == "07_PopulationCounts":
        cells = pal.NewPopGrid((stage.SIZE, stage.SIZE))
        for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
            cells[x, y] = 1
        multi = pal.NewMultinomial()
        nutrient = pal.NewPDEgrid((stage.SIZE, stage.SIZE))
        nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
        nutrient[:] = 1.0
        stage.Step(cells, multi)
        stage.StepNutrient(cells, nutrient)
        assert cells.GetPop() == 4
        assert np.isfinite(np.asarray(nutrient[:])).all()
        return

    cells = stage.Setup()
    pix, window = pal.StartPixWindow(stage.SIZE, stage.SIZE, scale=1, headless=True)
    try:
        stage.Draw(cells, pix)
        window.Update()
        if name in NAMES[2:]:
            empty = pal.NewIList()
            assert len(stage.EmptyNeighbors(cells, cells.All()[0], empty)) > 0
            if name in NAMES[3:]:
                stage.Step(cells, empty)
                assert len(cells.All()) == 4
        if name == "06_DiffusingNutrient":
            nutrient = pal.NewPDEgrid((stage.SIZE, stage.SIZE))
            nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
            nutrient[:] = 1.0
            stage.StepNutrient(cells, nutrient)
            assert np.isfinite(np.asarray(nutrient[:])).all()
    finally:
        window.Close()
