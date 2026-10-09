import time
import PythonAutomataLibrary as pal

# pal.FastMode()

X=100
Y=100
VIS_SCALE=5
TUMOR_RAD=10
PAUSE=0.005
TIMESTEPS=10000
RESISTANT_PROB=0.5

TIMESTEP=2.0/24
SPACE_STEP=20


def ProbScale(prob,timeScale):
    return 1-(1-prob)**timeScale


DIV_PROB_SEN=ProbScale(0.5,TIMESTEP)
DIV_PROB_RES=ProbScale(0.2,TIMESTEP)
DEATH_PROB=ProbScale(0.02,TIMESTEP)
DRUG_DEATH=ProbScale(0.8,TIMESTEP)

DRUG_START=20/TIMESTEP
DRUG_PERIOD=15/TIMESTEP
DRUG_DURATION=2/TIMESTEP

DRUG_UPTAKE=-0.03*TIMESTEP
DRUG_DIFF_RATE=0.02*60*60*24*(TIMESTEP/(SPACE_STEP*SPACE_STEP))
DRUG_BOUNDARY_VAL=1.0

RESISTANT=pal.RGB(0,255,0)
SENSITIVE=pal.RGB(0,0,255)

TYPE=0

DIV_HOOD=pal.MooreHood(2,excludeCenter=True)
TUMOR_HOOD=pal.CircleHood(2,rad=TUMOR_RAD)

HeatmapRGB=(pal.RGB(0,0,0),pal.RGB(255,0,0),pal.RGB(255,255,0),pal.RGB(255,255,255))


@pal.njit(cache=True)
def InitTumor(grid:pal.AgentGrid,resistantProb):
    for x,y in grid.Hood(TUMOR_HOOD,grid.xDim//2,grid.yDim//2):
        cell=grid.NewAgentSQ(x,y)

        if pal.Random()<resistantProb:
            grid[cell,TYPE]=RESISTANT
        else:
            grid[cell,TYPE]=SENSITIVE


@pal.njit(cache=True)
def ModelStep(grid:pal.AgentGrid,drug:pal.PDEgrid,empty:pal.IList,tick,drugDuration):
    for cell in grid.All(shuffle=True):
        i=grid.I(cell)

        # Drug uptake
        drug.Add(drug[i]*DRUG_UPTAKE,i)

        # Chance of death
        if grid[cell,TYPE]==RESISTANT:
            deathProb=DEATH_PROB
        else:
            deathProb=DEATH_PROB+drug[i]*DRUG_DEATH

        if pal.Random()<deathProb:
            grid.Dispose(cell)
            continue

        # Chance of division
        if grid[cell,TYPE]==RESISTANT:
            divProb=DIV_PROB_RES
        else:
            divProb=DIV_PROB_SEN

        if pal.Random()<divProb:
            empty.Clear()

            for x,y in grid.Hood(DIV_HOOD,grid.XSQ(cell),grid.YSQ(cell)):
                if grid.counts[x,y]==0:
                    empty.Append(grid.ToI(x,y))

            if len(empty)>0:
                daughter=grid.NewAgentSQ(empty.Random())
                grid[daughter,TYPE]=grid[cell,TYPE]

    # Drug treatment schedule
    periodTick=(tick-DRUG_START)%DRUG_PERIOD

    if periodTick>0 and periodTick<drugDuration:
        drug.DiffusionADI(DRUG_DIFF_RATE,DRUG_BOUNDARY_VAL,DRUG_BOUNDARY_VAL,DRUG_BOUNDARY_VAL,DRUG_BOUNDARY_VAL)
    else:
        drug.DiffusionADI(DRUG_DIFF_RATE)

    drug.Update()


@pal.njit(cache=True)
def DrawModel(grid:pal.AgentGrid,drug:pal.PDEgrid,pix:pal.Pix,xOffset):
    for x in range(X):
        for y in range(Y):
            i=grid.ToI(x,y)

            if grid.counts[i]:
                for cell in grid.AgentsAt(i):
                    pix[x+xOffset,y]=int(grid[cell,TYPE])
                    break
            else:
                pix[x+xOffset,y]=pal.ColorScale(HeatmapRGB,drug[i])

def main():
    pix,win=pal.StartPixWindow(
        X*3,
        Y,
        scale=VIS_SCALE,
        title="Competitive Release"
    )

    grids:list[pal.AgentGrid]=[
        pal.NewAgentGrid((X,Y),numAgentProps=1),
        pal.NewAgentGrid((X,Y),numAgentProps=1),
        pal.NewAgentGrid((X,Y),numAgentProps=1),
    ]

    drugs:list[pal.PDEgrid]=[
        pal.NewPDEgrid((X,Y)),
        pal.NewPDEgrid((X,Y)),
        pal.NewPDEgrid((X,Y)),
    ]

    empties:list[pal.IList]=[
        pal.NewIList(),
        pal.NewIList(),
        pal.NewIList(),
    ]

    for grid in grids:
        InitTumor(grid,RESISTANT_PROB)

    # No drug, constant drug, intermittent drug
    drugDurations=[
        0,
        200,
        DRUG_DURATION,
    ]

    pal.AwaitWindows()

    with open("populations.csv","w") as popsOut:
        for tick in range(TIMESTEPS):
            if not win.IsOpen():
                break

            time.sleep(PAUSE)

            for i in range(3):
                ModelStep(
                    grids[i],
                    drugs[i],
                    empties[i],
                    tick,
                    drugDurations[i]
                )
                DrawModel(
                    grids[i],
                    drugs[i],
                    pix,
                    i*X
                )

            win.Update()

            popsOut.write(
                f"{grids[0].GetPop()},{grids[1].GetPop()},{grids[2].GetPop()}\n"
            )

            if tick%int(10/TIMESTEP)==0:
                win.Save(f"ModelsDay{tick*TIMESTEP}.png")


if __name__=="__main__":
    main()