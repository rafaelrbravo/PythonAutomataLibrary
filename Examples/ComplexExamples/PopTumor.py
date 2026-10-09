import PythonAutomataLibrary as pal

TUMOR=0
M1=1
M2=2
MBOTH=3
N_TYPES=4

SIDE_LEN=150
LATTICE_CAP=100000

DIV_RATE=0.010
DIV_RATE_MUT=DIV_RATE+0.01
DIV_RATE_DOUBLE_MUT=DIV_RATE+0.02

DEATH_RATE=0.001
MIGRATION_RATE=0.008

MUT_PROB1=2e-6
MUT_PROB2=2e-6

INIT_POP=1000
TIMESTEPS=4001

WHITE=pal.RGB(255,255,255)

VON_NEUMANN=pal.VonNeumannHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Init(cells,sums):
    x=cells[TUMOR].xDim//2
    y=cells[TUMOR].yDim//2

    cells[TUMOR].Add(INIT_POP,x,y)
    cells[TUMOR].Update()

    sums[x,y]=INIT_POP


@pal.njit(cache=True)
def UpdateSums(cells,sums):
    for i in range(len(sums)):
        sums[i]=(
            cells[TUMOR][i]
            +cells[M1][i]
            +cells[M2][i]
            +cells[MBOTH][i]
        )


@pal.njit(cache=True)
def StepType(cells,sums,mn,cellType):
    grid=cells[cellType]

    for i in range(len(grid)):
        ct=grid[i]

        if ct==0:
            continue

        density=sums[i]/LATTICE_CAP

        if cellType==TUMOR:
            divRate=DIV_RATE
        elif cellType==MBOTH:
            divRate=DIV_RATE_DOUBLE_MUT
        else:
            divRate=DIV_RATE_MUT

        divProb=divRate*(1.0-density)

        if divProb<0:
            divProb=0

        mn.Setup(ct)

        ctDiv=mn.Sample(divProb)
        ctDie=mn.Sample(DEATH_RATE)

        ctM1mut=0
        ctM2mut=0
        ctBothmut=0

        if cellType==TUMOR:
            mn.Setup(ctDiv)

            ctM1mut=mn.Sample(MUT_PROB1)
            ctM2mut=mn.Sample(MUT_PROB2)

        elif cellType==M1:
            ctBothmut=mn.Binomial(ctDiv,MUT_PROB2)

        elif cellType==M2:
            ctBothmut=mn.Binomial(ctDiv,MUT_PROB1)

        grid.Add(
            ctDiv
            -ctDie
            -ctM1mut
            -ctM2mut
            -ctBothmut,
            i
        )

        if ctM1mut:
            cells[M1].Add(ctM1mut,i)

        if ctM2mut:
            cells[M2].Add(ctM2mut,i)

        if ctBothmut:
            cells[MBOTH].Add(ctBothmut,i)


@pal.njit(cache=True)
def Migrate(grid:pal.PopGrid,mn:pal.Multinomial):
    for x in range(grid.xDim):
        for y in range(grid.yDim):
            ct=grid[x,y]

            if ct==0:
                continue

            # Select the individuals that attempt migration.
            moving=mn.Binomial(ct,MIGRATION_RATE)

            if moving==0:
                continue

            # Uniformly distribute migrants among the four directions.
            mn.Setup(moving)

            left=mn.Sample(0.25)
            right=mn.Sample(0.25)
            down=mn.Sample(0.25)
            up=mn.Sample(0.25)

            moved=0

            if x>0:
                grid.Add(left,x-1,y)
                moved+=left

            if x<grid.xDim-1:
                grid.Add(right,x+1,y)
                moved+=right

            if y>0:
                grid.Add(down,x,y-1)
                moved+=down

            if y<grid.yDim-1:
                grid.Add(up,x,y+1)
                moved+=up

            # Only remove migrants whose destination was in bounds.
            grid.Add(-moved,x,y)


@pal.njit(cache=True)
def Step(cells,sums,mn):
    for cellType in range(N_TYPES):
        StepType(cells,sums,mn,cellType)

    for cellType in range(N_TYPES):
        cells[cellType].Update()

    for cellType in range(N_TYPES):
        Migrate(cells[cellType],mn)
        cells[cellType].Update()

    UpdateSums(cells,sums)


@pal.njit(cache=True)
def Draw(cells,pix,xOffset):
    for x in range(SIDE_LEN):
        for y in range(SIDE_LEN):
            tumor=cells[TUMOR][x,y]/LATTICE_CAP
            m1=cells[M1][x,y]/LATTICE_CAP
            m2=cells[M2][x,y]/LATTICE_CAP
            both=cells[MBOTH][x,y]/LATTICE_CAP

            r=min(1.0,tumor+m1)
            g=min(1.0,tumor+m2)
            b=min(1.0,tumor+both)

            pix[x+xOffset,y]=pal.RGB(
                int(r*255),
                int(g*255),
                int(b*255)
            )


@pal.njit(cache=True)
def Population(cells):
    return (
        cells[TUMOR].GetPop()
        +cells[M1].GetPop()
        +cells[M2].GetPop()
        +cells[MBOTH].GetPop()
    )


def main():
    # pal.FastMode()

    cells=(
        pal.NewPopGrid(dimensions=(SIDE_LEN,SIDE_LEN)),
        pal.NewPopGrid(dimensions=(SIDE_LEN,SIDE_LEN)),
        pal.NewPopGrid(dimensions=(SIDE_LEN,SIDE_LEN)),
        pal.NewPopGrid(dimensions=(SIDE_LEN,SIDE_LEN))
    )

    sums=pal.NewGrid(
        dimensions=(SIDE_LEN,SIDE_LEN),
        dtype="int64"
    )

    mn=pal.NewMultinomial()

    pix,win=pal.StartPixWindow(
        xDim=SIDE_LEN*4,
        yDim=SIDE_LEN,
        scale=2,
        title="Population Tumor"
    )

    for i in range(len(pix)):
        pix[i]=WHITE

    Init(cells,sums)

    pal.AwaitWindows()

    for tick in range(TIMESTEPS):
        if not win.IsOpen():
            break
        
        Step(cells,sums,mn)
    
        if (tick+1)%1000==0:
            snapshot=(tick+1)//1000-1
    
            print(Population(cells))
    
            Draw(cells,pix,snapshot*SIDE_LEN)
            win.Update()
    
            Step(cells,sums,mn)
    
        win.Save("PopulationGrid.png")


if __name__=="__main__":
    main()