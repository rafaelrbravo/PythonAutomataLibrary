import time
import PythonAutomataLibrary as pal

X=100
Y=100
SCALE=10
DEATH_PROB=0.01
BIRTH_PROB=0.2

RED=pal.RGB(255,0,0)
BLACK=pal.RGB(0,0,0)

MOORE=pal.MooreHood(2,excludeCenter=True)
INIT_HOOD=pal.CircleHood(2,rad=10)


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for x,y in grid.Hood(INIT_HOOD,grid.xDim//2,grid.yDim//2):
        grid.NewAgentSQ(x,y)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,empty:pal.IList):
    # Snapshot + shuffle means each agent present at the start of the timestep
    # acts once, while newborn agents wait until the next timestep.
    for agent in grid.All(shuffle=True):
        if pal.Random()<DEATH_PROB:
            grid.Dispose(agent)
            continue

        if pal.Random()<BIRTH_PROB:
            empty.Clear()

            for x,y in grid.Hood(MOORE,grid.XSQ(agent),grid.YSQ(agent),unroll=True):
                if grid.counts[x,y]==0:
                    empty.Append(grid.ToI(x,y))

            if len(empty):
                grid.NewAgentSQ(empty.Random())


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,pix:pal.Pix):
    pix[:]=BLACK

    for agent in grid.All():
        pix[grid.I(agent)]=RED


def main():
    pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE)
    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=0,isStackable=False)
    empty=pal.NewIList()

    Setup(grid)

    pal.AwaitWindows()
    while win.IsOpen():
        Step(grid,empty)
        Draw(grid,pix)
        win.Update()
        time.sleep(0.01)


if __name__=="__main__":
    main()