import PythonAutomataLibrary as pal

X=1000
Y=1000
SCALE=1
TIMESTEPS=100000

DIV_PROB=0.2
MUT_PROB=0.001
DIE_PROB=0.1
MUT_ADVANTAGE=1.08
MAX_MUTATIONS=19
INIT_RADIUS=10

N_MUTATIONS=0
R=1
G=2
B=3

BLACK=pal.RGB(0,0,0)

VON_NEUMANN=pal.VonNeumannHood(2,excludeCenter=True)
INIT_HOOD=pal.CircleHood(2,rad=INIT_RADIUS)


@pal.njit(cache=True)
def MutateColor(value):
    value+=pal.RandInt(81)-40

    if value<0:
        return 0
    if value>255:
        return 255

    return value


@pal.njit(cache=True)
def DrawCell(grid:pal.AgentGrid,pix:pal.Pix,cell):
    pix[grid.I(cell)]=pal.RGB(
        int(grid[cell,R]),
        int(grid[cell,G]),
        int(grid[cell,B])
    )


@pal.njit(cache=True)
def PossiblyMutate(grid:pal.AgentGrid,pix:pal.Pix,cell):
    if grid[cell,N_MUTATIONS]<MAX_MUTATIONS and pal.Random()<MUT_PROB:
        grid[cell,N_MUTATIONS]+=1
        grid[cell,R]=MutateColor(grid[cell,R])
        grid[cell,G]=MutateColor(grid[cell,G])
        grid[cell,B]=MutateColor(grid[cell,B])

        DrawCell(grid,pix,cell)


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid,pix:pal.Pix):
    for i in range(len(grid)):
        pix[i]=BLACK

    cx=grid.xDim//2
    cy=grid.yDim//2

    for x,y in grid.Hood(INIT_HOOD,cx,cy):
        cell=grid.NewAgentSQ(x,y)

        grid[cell,N_MUTATIONS]=1
        grid[cell,R]=MutateColor(128)
        grid[cell,G]=MutateColor(128)
        grid[cell,B]=MutateColor(128)

        DrawCell(grid,pix,cell)


@pal.njit(cache=True)
def Divide(grid:pal.AgentGrid,pix:pal.Pix,cell):
    x=grid.XSQ(cell)
    y=grid.YSQ(cell)

    nEmpty=0
    daughterX=0
    daughterY=0

    # Uniformly select one empty neighbor without allocating an IList.
    for nx,ny in grid.Hood(VON_NEUMANN,x,y,unroll=True):
        if grid.counts[nx,ny]==0:
            nEmpty+=1

            if pal.RandInt(nEmpty)==0:
                daughterX=nx
                daughterY=ny

    if nEmpty==0:
        return

    daughter=grid.NewAgentSQ(daughterX,daughterY)

    grid[daughter,N_MUTATIONS]=grid[cell,N_MUTATIONS]
    grid[daughter,R]=grid[cell,R]
    grid[daughter,G]=grid[cell,G]
    grid[daughter,B]=grid[cell,B]

    DrawCell(grid,pix,daughter)

    PossiblyMutate(grid,pix,cell)
    PossiblyMutate(grid,pix,daughter)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,pix:pal.Pix):
    for cell in grid.All(shuffle=True):
        if pal.Random()<DIE_PROB:
            pix[grid.I(cell)]=BLACK
            grid.Dispose(cell)
            continue

        if pal.Random()<DIV_PROB*MUT_ADVANTAGE**grid[cell,N_MUTATIONS]:
            Divide(grid,pix,cell)


def main():
    # pal.FastMode()

    grid=pal.NewAgentGrid(
        dimensions=(X,Y),
        numAgentProps=4,
        isStackable=False
    )

    pix,win=pal.StartPixWindow(
        xDim=X,
        yDim=Y,
        scale=SCALE,
        title="Phylogeny Tracker"
    )

    Setup(grid,pix)
    win.Update()

    pal.AwaitWindows()

    for _ in range(TIMESTEPS):
        if not win.IsOpen():
            break

        Step(grid,pix)
        win.Update()


if __name__=="__main__":
    main()