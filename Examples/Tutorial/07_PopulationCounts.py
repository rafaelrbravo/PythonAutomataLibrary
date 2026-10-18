import time
import PythonAutomataLibrary as pal

SIZE = 30
BLACK = pal.RGB(0, 0, 0)
RED = pal.RGB(255, 0, 0)
NUTRIENT_COLORS = (BLACK, pal.RGB(255, 255, 255))

MOORE = pal.MooreHood(2, excludeCenter=True)

@pal.njit(cache=True)
def Step(cells: pal.PopGrid, multi: pal.Multinomial):
    for i in cells.All():
        multi.Setup(cells[i])
        x, y = cells.ItoX(i), cells.ItoY(i)
        for nx, ny in cells.Hood(MOORE, x, y, unroll=True):
            moved = multi.Sample(1.0 / 8)
            cells.Add(-moved, i)
            cells.Add(moved, nx, ny)
    cells.Update()

@pal.njit(cache=True)
def StepNutrient(cells: pal.PopGrid, nutrient: pal.PDEgrid):
    for i in range(len(cells)):
        nutrient[i] = max(0.0, nutrient[i] - 0.2 * cells[i])
    nutrient.Diffusion(0.1, xMinBC=1.0, xMaxBC=1.0, yMinBC=1.0, yMaxBC=1.0)
    nutrient.Update()

@pal.njit(cache=True)
def Draw(cells: pal.PopGrid, pix: pal.Pix):
    pix[:] = BLACK
    for i in range(len(cells)):
        if cells[i]:
            pix[i] = RED

def DrawNutrient(nutrient, pix):
    for i in range(len(nutrient)):
        pix[i] = pal.ColorScale(NUTRIENT_COLORS, max(0.0, min(1.0, nutrient[i])))

def main():
    pal.Seed(1234)
    cells = pal.NewPopGrid((SIZE, SIZE))
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells[x, y] = 1
    initialPop = cells.GetPop()
    multi = pal.NewMultinomial()
    nutrient = pal.NewPDEgrid((SIZE, SIZE))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    pix, window = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Population random walk")
    nutrientPix, nutrientWindow = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Nutrient")
    DrawNutrient(nutrient, nutrientPix)
    nutrientWindow.Update()
    Draw(cells, pix)
    window.Update()
    pal.AwaitWindows()
    while window.IsOpen() and nutrientWindow.IsOpen():
        Step(cells, multi)
        assert cells.GetPop() == initialPop  # movement conserves cell count
        StepNutrient(cells, nutrient)
        Draw(cells, pix)
        window.Update()
        DrawNutrient(nutrient, nutrientPix)
        nutrientWindow.Update()
        time.sleep(0.05)
    window.Close()
    nutrientWindow.Close()

if __name__ == "__main__":
    main()
