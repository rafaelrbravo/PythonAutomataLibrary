import time
import PythonAutomataLibrary as pal

SIZE = 30
BLACK = pal.RGB(0, 0, 0)
RED = pal.RGB(255, 0, 0)

def Draw(cells, pix):
    pix[:] = BLACK
    for agent in cells.All():
        pix[cells.XSQ(agent), cells.YSQ(agent)] = RED

def Setup(cells):
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
@pal.njit(cache=True)
def EmptyNeighbors(cells: pal.AgentGrid, agent: int, empty: pal.IList, moore):
    empty.Clear()
    for x, y in cells.Hood(moore, cells.XSQ(agent), cells.YSQ(agent), unroll=True):
        if cells.counts[x, y] == 0:
            empty.Append(cells.ToI(x, y))
    return empty

@pal.njit(cache=True)
def Step(cells: pal.AgentGrid, empty: pal.IList, moore):
    for agent in cells.All(shuffle=True):
        EmptyNeighbors(cells, agent, empty, moore)
        if len(empty):
            site = empty.Random()
            cells.MoveSQ(agent, cells.ItoX(site), cells.ItoY(site))

def main():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    Setup(cells)
    empty = pal.NewIList()
    moore = pal.MooreHood(2, excludeCenter=True)
    pix, window = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Cells")
    Draw(cells, pix)
    window.Update()
    pal.AwaitWindows()
    while window.IsOpen():
        Step(cells, empty, moore)
        Draw(cells, pix)
        window.Update()
        time.sleep(0.05)
    window.Close()

if __name__ == "__main__":
    main()
