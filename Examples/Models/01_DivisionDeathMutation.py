import PythonAutomataLibrary as pal

# pal.FastMode()

X=1000
Y=1000
SCALE=1
TIMESTEPS=2000

DIV_PROB=0.2
MUT_PROB=0.003
DIE_PROB=0.1
MUT_ADVANTAGE=1.08
MAX_MUTATIONS=19

N_MUTATIONS=0

BLACK=pal.RGB(0,0,0)
MUT_COLORS=(pal.RGB(0,0,128),pal.RGB(0,0,255),pal.RGB(0,255,255),pal.RGB(255,255,0),pal.RGB(255,0,0),pal.RGB(128,0,0))

HOOD=pal.VonNeumannHood(2,excludeCenter=True)
INIT_HOOD=pal.CircleHood(2,rad=5)


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for x,y in grid.Hood(INIT_HOOD,grid.xDim//2,grid.yDim//2):
        agent=grid.NewAgentSQ(x,y)
        grid[agent,N_MUTATIONS]=0


@pal.njit(cache=True)
def Mutate(grid:pal.AgentGrid,agent):
    if grid[agent,N_MUTATIONS]<MAX_MUTATIONS and pal.Random()<MUT_PROB:
        grid[agent,N_MUTATIONS]+=1


@pal.njit(cache=True)
def Divide(grid:pal.AgentGrid,agent,empty:pal.IList):
    empty.Clear()

    for x,y in grid.Hood(HOOD,grid.XSQ(agent),grid.YSQ(agent),unroll=True):
        if grid.counts[x,y]==0:
            empty.Append(grid.ToI(x,y))

    if len(empty):
        daughter=grid.NewAgentSQ(empty.Random())
        grid[daughter,N_MUTATIONS]=grid[agent,N_MUTATIONS]

        Mutate(grid,agent)
        Mutate(grid,daughter)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,empty:pal.IList):
    for agent in grid.All(shuffle=True):
        if pal.Random()<DIE_PROB:
            grid.Dispose(agent)
        elif pal.Random()<DIV_PROB*MUT_ADVANTAGE**grid[agent,N_MUTATIONS]:
            Divide(grid,agent,empty)


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,pix:pal.Pix):
    pix[:]=BLACK

    for agent in grid.All():
        mutationCt=grid[agent,N_MUTATIONS]
        pix[grid.I(agent)]=pal.ColorScale(MUT_COLORS,mutationCt/MAX_MUTATIONS)


def main():
    pal.Seed(1)
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Division Death Mutation")

    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=1,isStackable=False)
    empty=pal.NewIList()

    Setup(grid)
    Draw(grid,pix)
    win.Update()

    pal.AwaitWindows()

    for _ in range(TIMESTEPS):
        Step(grid,empty)
        Draw(grid,pix)
        win.Update()


if __name__=="__main__":
    main()