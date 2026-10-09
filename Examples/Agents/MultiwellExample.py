import time
import PythonAutomataLibrary as pal

X=100
Y=100
WELLS_X=3
WELLS_Y=2
SPACING=1
SCALE=5
DEATH_PROB=0.01
BIRTH_PROB=0.2

COLORS=(
    pal.RGB(255,0,0),
    pal.RGB(0,255,0),
    pal.RGB(0,0,255),
    pal.RGB(255,255,0),
    pal.RGB(0,255,255),
    pal.RGB(255,0,255)
)
BLACK=pal.RGB(0,0,0)
WHITE=pal.RGB(255,255,255)

COLOR=0

INIT_HOOD=pal.CircleHood(2,rad=2)
MOORE=pal.MooreHood(2,excludeCenter=True)


@pal.njit(cache=True)
def SetupWell(grid:pal.AgentGrid,color):
    for x,y in grid.Hood(INIT_HOOD,grid.xDim//2,grid.yDim//2):
        agent=grid.NewAgentSQ(x,y)
        grid[agent,COLOR]=color


@pal.njit(cache=True)
def StepWell(grid:pal.AgentGrid,empty:pal.IList):
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
                child=grid.NewAgentSQ(empty.Random())
                grid[child,COLOR]=grid[agent,COLOR]


@pal.njit(cache=True)
def DrawWell(grid:pal.AgentGrid,pix:pal.Pix,wellX,wellY):
    xOffset=wellX*(X+SPACING)
    yOffset=wellY*(Y+SPACING)

    pix[xOffset:xOffset+X,yOffset:yOffset+Y]=WHITE

    for agent in grid.All():
        pix[xOffset+grid.XSQ(agent),yOffset+grid.YSQ(agent)]=grid[agent,COLOR]


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=WELLS_X*X+(WELLS_X-1)*SPACING,yDim=WELLS_Y*Y+(WELLS_Y-1)*SPACING,scale=SCALE)
    pix[:]=BLACK

    grids=[
        pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=1,isStackable=False)
        for _ in COLORS
    ]
    empty=[pal.NewIList() for _ in COLORS]

    for grid,color in zip(grids,COLORS):
        SetupWell(grid,color)

    pal.AwaitWindows()
    while win.IsOpen():
        for i,grid in enumerate(grids):
            StepWell(grid,empty[i])
            DrawWell(grid,pix,i%WELLS_X,i//WELLS_X)

        win.Update()
        time.sleep(0.01)


if __name__=="__main__":
    main()