import pickle
import time
import PythonAutomataLibrary as pal

X=100
Y=100
SCALE=10
DEATH_PROB=0.01
BIRTH_PROB=0.2

RED=pal.RGB(255,0,0)
BLACK=pal.RGB(0,0,0)

INIT_HOOD=pal.CircleHood(2,rad=10)
MOORE=pal.MooreHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for x,y in grid.Hood(INIT_HOOD,grid.xDim//2,grid.yDim//2):
        grid.NewAgentSQ(x,y)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,empty:pal.IList):
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
        pix[grid.XSQ(agent),grid.YSQ(agent)]=RED


def main():
    # pal.FastMode()

    # The saved state can contain PAL components alongside ordinary Python state.
    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE)

    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=0,isStackable=False)
    empty=pal.NewIList()

    step=0
    Setup(grid)
    savedState=None
    loads=0

    pal.AwaitWindows()
    while win.IsOpen():
        if step==100:
            savedState=pickle.dumps({"grid":grid,"empty":empty,"step":step})

        if step==200 and savedState is not None and loads<3:
            state=pickle.loads(savedState)
            grid=state["grid"]
            empty=state["empty"]
            step=state["step"]
            loads+=1

        Step(grid,empty)
        Draw(grid,pix)
        win.Update()
        time.sleep(0.01)
        step+=1


if __name__=="__main__":
    main()