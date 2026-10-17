import time
import PythonAutomataLibrary as pal

SIZE = 30
BLACK = pal.RGB(0, 0, 0)
RED = pal.RGB(255, 0, 0)

def Draw(cells, pix):
    pix[:] = BLACK
    for agent in cells.All():
        pix[cells.XSQ(agent), cells.YSQ(agent)] = RED

def Setup():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
    return cells

MOORE = pal.MooreHood(2, excludeCenter=True)

def EmptyNeighbors(cells, agent, empty):
    empty.Clear()
    for x, y in cells.Hood(MOORE, cells.XSQ(agent), cells.YSQ(agent)):
        if cells.counts[x, y] == 0:
            empty.Append(cells.ToI(x, y))
    return empty

def main():
    cells = Setup()
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
