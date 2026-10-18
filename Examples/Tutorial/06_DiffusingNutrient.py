import time
import PythonAutomataLibrary as pal

SIZE = 30
BLACK = pal.RGB(0, 0, 0)
RED = pal.RGB(255, 0, 0)
NUTRIENT_COLORS = (BLACK, pal.RGB(255, 255, 255))

MOORE = pal.MooreHood(2, excludeCenter=True)

def Draw(cells, pix):
    pix[:] = BLACK
    for agent in cells.All():
        pix[cells.XSQ(agent), cells.YSQ(agent)] = RED

def Setup(cells):
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
@pal.njit(cache=True)
def EmptyNeighbors(cells: pal.AgentGrid, agent: int, empty: pal.IList):
    empty.Clear()
    for x, y in cells.Hood(MOORE, cells.XSQ(agent), cells.YSQ(agent), unroll=True):
        if cells.counts[x, y] == 0:
            empty.Append(cells.ToI(x, y))
    return empty

@pal.njit(cache=True)
def Step(cells: pal.AgentGrid, empty: pal.IList):
    for agent in cells.All(shuffle=True):
        EmptyNeighbors(cells, agent, empty)
        if len(empty):
            site = empty.Random()
            cells.MoveSQ(agent, cells.ItoX(site), cells.ItoY(site))

def StepNutrient(cells, nutrient):
    for agent in cells.All():
        x, y = cells.XSQ(agent), cells.YSQ(agent)
        nutrient[x, y] = max(0.0, nutrient[x, y] - 0.01)
    nutrient.Diffusion(0.1, xMinBC=1.0, xMaxBC=1.0, yMinBC=1.0, yMaxBC=1.0)
    nutrient.Update()

def DrawNutrient(nutrient, pix):
    for x in range(SIZE):
        for y in range(SIZE):
            pix[x, y] = pal.ColorScale(NUTRIENT_COLORS, max(0.0, min(1.0, nutrient[x, y])))

def main():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    Setup(cells)
    nutrient = pal.NewPDEgrid((SIZE, SIZE))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    empty = pal.NewIList()
    pix, window = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Cells")
    Draw(cells, pix)
    window.Update()
    nutrientPix, nutrientWindow = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Nutrient")
    DrawNutrient(nutrient, nutrientPix)
    nutrientWindow.Update()
    pal.AwaitWindows()
    while window.IsOpen() and nutrientWindow.IsOpen():
        Step(cells, empty)
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
