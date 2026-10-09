import PythonAutomataLibrary as pal


X=80
Y=20
Z=80

VESSEL_SPACING=15
VESSEL_MIG_PROB=0.8
INIT_DIFF_STEPS=100

DIFF_RATE=0.5/6
TUMOR_METABOLISM_RATE=-0.04
NORMAL_METABOLISM_RATE=-0.005
VESSEL_CONC=1.0
DEATH_CONC=0.01

TYPE=0

VESSEL=0
TUMOR=1

BACKGROUND_COLOR=pal.RGB(38,1,5)
VESSEL_COLOR=pal.RGB(255,78,68)

SPACING_HOOD=pal.CircleHood(2,rad=VESSEL_SPACING)

VN=pal.VonNeumannHood(3,excludeCenter=True)

VESSEL_MOVE=(
    (1,0,0),
    (-1,0,0),
    (0,0,1),
    (0,0,-1),
)


@pal.njit(cache=True)
def NewVessel(grid:pal.AgentGrid,x,y,z):
    cell=grid.NewAgentSQ(x,y,z)
    grid[cell,TYPE]=VESSEL


@pal.njit(cache=True)
def NewTumor(grid:pal.AgentGrid,i):
    cell=grid.NewAgentSQ(i)
    grid[cell,TYPE]=TUMOR


@pal.njit(cache=True)
def DiffStep(grid:pal.AgentGrid,oxygen:pal.PDEgrid):
    for cell in grid.All():
        i=grid.I(cell)

        if grid[cell,TYPE]==VESSEL:
            oxygen[i]=VESSEL_CONC
        else:
            oxygen.Add(oxygen[i]*TUMOR_METABOLISM_RATE,i)

    for i in range(len(oxygen)):
        oxygen.Add(oxygen[i]*NORMAL_METABOLISM_RATE,i)

    oxygen.Diffusion(DIFF_RATE)
    oxygen.Update()


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,oxygen:pal.PDEgrid,empty:pal.IList):
    tumorCt=0

    for cell in grid.All(shuffle=True):
        if grid[cell,TYPE]!=TUMOR:
            continue

        tumorCt+=1
        i=grid.I(cell)
        conc=oxygen[i]

        if conc<DEATH_CONC and pal.Random()<1.0-conc/DEATH_CONC:
            grid.Dispose(cell)
            continue

        if pal.Random()<conc:
            empty.Clear()

            for x,y,z in grid.Hood(
                VN,
                grid.XSQ(cell),
                grid.YSQ(cell),
                grid.ZSQ(cell),
                unroll=True,
            ):
                if grid.counts[x,y,z]==0:
                    empty.Append(grid.ToI(x,y,z))

            if len(empty):
                NewTumor(grid,empty.Random())

    if tumorCt==0:
        i=pal.RandInt(len(grid))
        if grid.counts[i]==0:
            NewTumor(grid,i)

    DiffStep(grid,oxygen)


@pal.njit(cache=True)
def GenVessel(grid:pal.AgentGrid,x,z,migProb):
    for y in range(grid.yDim):
        if pal.Random()<migProb:
            options=0

            for nx,ny,nz in grid.Hood(VESSEL_MOVE,x,y,z,unroll=True):
                options+=1
                if pal.RandInt(options)==0:
                    x=nx
                    z=nz

        for occupant in grid.AgentsAt(x,y,z):
            grid.Dispose(occupant)

        NewVessel(grid,x,y,z)


@pal.njit(cache=True)
def GenVessels(grid:pal.AgentGrid,openSpots:pal.Grid,migProb):
    vesselCt=0

    for i in range(len(openSpots)):
        if openSpots[i]!=0:
            continue

        x=openSpots.ItoX(i)
        z=openSpots.ItoY(i)

        GenVessel(grid,x,z,migProb)
        vesselCt+=1

        for hx,hz in openSpots.Hood(SPACING_HOOD,x,z):
            openSpots[hx,hz]=1

    return vesselCt


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,oxygen:pal.PDEgrid,draw:pal.OpenGLDraw):
    draw.Clear()

   # Oxygen concentration on the far XZ plane.
    for x in range(X):
        for z in range(Z):
            conc=oxygen[x,Y-1,z]
            color=pal.ColorScale(
                (
                    pal.RGB(0,0,0),
                    pal.RGB(0,0,255),
                    pal.RGB(0,255,255),
                    pal.RGB(255,255,255),
                ),
                conc,
            )
            draw.Box(1.0,color,x+0.5,Y,z+0.5,yLen=0.0,zLen=1.0)

    for cell in grid.All():
        x=grid.X(cell)
        y=grid.Y(cell)
        z=grid.Z(cell)

        if grid[cell,TYPE]==VESSEL:
            draw.Circle(0.5,VESSEL_COLOR,x,y,z)
        else:
            conc=oxygen[grid.I(cell)]
            color=pal.ColorScale(
                (
                    pal.RGB(0,0,255),
                    pal.RGB(255,0,0),
                    pal.RGB(0,255,0),
                ),
                conc**0.5*0.8+0.2,
            )
            draw.Circle(0.3,color,x,y,z)


def main():
    # pal.FastMode()

    pal.Seed(0)

    draw,win=pal.StartOpenGLWindow(
        xDim=X,
        yDim=Y,
        zDim=Z,
        width=1000,
        height=1000,
        title="TumorVis",
    )

    grid=pal.NewAgentGrid(
        dimensions=(X,Y,Z),
        numAgentProps=1,
        isStackable=False,
    )
    oxygen=pal.NewPDEgrid(dimensions=(X,Y,Z))
    openSpots=pal.NewGrid(dimensions=(X,Z),dtype="int8")
    empty=pal.NewIList()

    GenVessels(grid,openSpots,VESSEL_MIG_PROB)

    for _ in range(INIT_DIFF_STEPS):
        DiffStep(grid,oxygen)

    pal.AwaitWindows()

    while win.IsOpen():
        Step(grid,oxygen,empty)
        Draw(grid,oxygen,draw)
        win.Update()


if __name__=="__main__":
    main()