import time
import PythonAutomataLibrary as pal

SIZE = 30
BLACK = pal.RGB(0, 0, 0)
RED = pal.RGB(255, 0, 0)

MOORE = pal.MooreHood(2, excludeCenter=True)

def Draw(cells: pal.AgentGrid, pix: pal.Pix):
    pix[:] = BLACK
    for agent in cells.All():
        pix[cells.I(agent)] = RED

def Setup(cells: pal.AgentGrid):
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
@pal.njit(cache=True)
def EmptyNeighbors(cells: pal.AgentGrid, agent: int, empty: pal.IList):
    empty.Clear()
    for x, y in cells.Hood(MOORE, cells.XSQ(agent), cells.YSQ(agent), unroll=True):
        if cells.counts[x, y] == 0:
            empty.Append(cells.ToI(x, y))

@pal.njit(cache=True)
def Step(cells: pal.AgentGrid, empty: pal.IList):
    for agent in cells.All(shuffle=True):
        EmptyNeighbors(cells, agent, empty)
        if len(empty):
            site = empty.Random()
            cells.MoveSQ(agent, cells.ItoX(site), cells.ItoY(site))

def main():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    Setup(cells)
    empty = pal.NewIList()
    pix, window = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Cells")
    Draw(cells, pix)
    window.Update()
    pal.AwaitWindows()
    while window.IsOpen():
        Step(cells, empty)
        Draw(cells, pix)
        window.Update()
        time.sleep(0.05)
    window.Close()

if __name__ == "__main__":
    main()
