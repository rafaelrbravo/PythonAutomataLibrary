"""Headless execution checks for the API Guide's progressive cell tutorial."""
import pickle
import numpy as np
import pytest

SIZE = 30
INITIAL = ((14, 14), (15, 14), (14, 15), (15, 15))


@pytest.fixture
def model(api):
    cells = api.NewAgentGrid((SIZE, SIZE), isStackable=False)
    for x, y in INITIAL:
        cells.NewAgentSQ(x, y)
    return cells


def test_stage_1_setup(model):
    assert len(model.All()) == 4


def test_stage_2_draw_data(api, model):
    # Test the Draw routine's pixel operations without opening a GUI window.
    pix = np.zeros((SIZE, SIZE), dtype=np.int32)
    black, red = api.RGB(0, 0, 0), api.RGB(255, 0, 0)
    pix[:] = black
    for agent in model.All():
        pix[model.XSQ(agent), model.YSQ(agent)] = red
    assert np.count_nonzero(pix == red) == 4


def test_stages_3_and_4_empty_neighbors_and_movement(api, model):
    hood = api.MooreHood(2, excludeCenter=True)
    empty = api.NewIList()

    def empty_neighbors(cells, agent, sites):
        sites.Clear()
        for x, y in cells.Hood(hood, cells.XSQ(agent), cells.YSQ(agent)):
            if cells.counts[x, y] == 0:
                sites.Append(cells.ToI(x, y))
        return sites

    first = model.All()[0]
    assert len(empty_neighbors(model, first, empty)) > 0
    for agent in model.All(shuffle=True):
        empty_neighbors(model, agent, empty)
        if len(empty):
            site = empty.Random()
            model.MoveSQ(agent, model.ItoX(site), model.ItoY(site))
    assert len(model.All()) == 4
    assert sum(int(model.counts[x, y]) for x in range(SIZE) for y in range(SIZE)) == 4


def test_stage_5_compiled_movement(api, model):
    hood = api.MooreHood(2, excludeCenter=True)

    @api.njit(cache=False)
    def empty_neighbors(cells, agent, sites):
        sites.Clear()
        for x, y in cells.Hood(hood, cells.XSQ(agent), cells.YSQ(agent)):
            if cells.counts[x, y] == 0:
                sites.Append(cells.ToI(x, y))
        return sites

    @api.njit(cache=False)
    def step(cells, empty):
        for agent in cells.All(shuffle=True):
            empty_neighbors(cells, agent, empty)
            if len(empty):
                site = empty.Random()
                cells.MoveSQ(agent, cells.ItoX(site), cells.ItoY(site))

    empty = api.NewIList()
    step(model, empty)
    assert len(model.All()) == 4


def test_stage_6_nutrient(api, model):
    nutrient = api.NewPDEgrid((SIZE, SIZE))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    for agent in model.All():
        x, y = model.XSQ(agent), model.YSQ(agent)
        nutrient[x, y] = max(0.0, nutrient[x, y] - 0.01)
    nutrient.Diffusion(0.1, xMinBC=1.0, xMaxBC=1.0, yMinBC=1.0, yMaxBC=1.0)
    nutrient.Update()
    assert 0.0 <= float(nutrient[14, 14]) < 1.0


def test_stage_7_popgrid_multinomial_nutrient(api):
    hood = api.MooreHood(2, excludeCenter=True)
    cells = api.NewPopGrid((SIZE, SIZE))
    for x, y in INITIAL:
        cells[x, y] = 1
    multi = api.NewMultinomial()

    @api.njit(cache=False)
    def step(cells, multi):
        for i in cells.All():
            multi.Setup(cells[i])
            x, y = cells.ItoX(i), cells.ItoY(i)
            for nx, ny in cells.Hood(hood, x, y):
                moved = multi.Sample(1.0 / 8)
                cells.Add(-moved, i)
                cells.Add(moved, nx, ny)
        cells.Update()

    step(cells, multi)
    assert cells.GetPop() == 4
    nutrient = api.NewPDEgrid((SIZE, SIZE))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    for i in range(len(cells)):
        nutrient[i] = max(0.0, nutrient[i] - 0.01 * cells[i])
    nutrient.Diffusion(0.1, xMinBC=1.0, xMaxBC=1.0, yMinBC=1.0, yMaxBC=1.0)
    nutrient.Update()
    assert np.isfinite(np.asarray(nutrient[:])).all()


def test_stage_8_checkpoint(api, model, tmp_path):
    nutrient = api.NewPDEgrid((SIZE, SIZE))
    step = 3
    path = tmp_path / "cells.pkl"
    with path.open("wb") as f:
        pickle.dump({"cells": model, "nutrient": nutrient, "step": step}, f)
    with path.open("rb") as f:
        state = pickle.load(f)
    assert len(state["cells"].All()) == 4
    assert state["step"] == 3
