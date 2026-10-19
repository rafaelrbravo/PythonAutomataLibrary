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

def EmptyNeighbors(cells: pal.AgentGrid, agent, empty: pal.IList):
    empty.Clear()
    for x, y in cells.Hood(MOORE, cells.XSQ(agent), cells.YSQ(agent)):
        if cells.counts[x, y] == 0:
            empty.Append(cells.ToI(x, y))
    return empty

def main():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    Setup(cells)
    empty = pal.NewIList()
    print(EmptyNeighbors(cells, cells.All()[0], empty).All())
    pix, window = pal.StartPixWindow(SIZE, SIZE, scale=12, title="Cells")
    Draw(cells, pix)
    window.Update()
    pal.AwaitWindows()
    while window.IsOpen():
        time.sleep(0.05)
    window.Close()

if __name__ == "__main__":
    main()
