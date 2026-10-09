import PythonAutomataLibrary as pal

# pal.FastMode()

X=200
Y=200
SCALE=5

DIV_PROB=1.0/24
DEATH_PROB=1.0/1000
STEM_DIV_PROB=7.0/10
MAX_DIVS=11

RUN_TICKS=20000

DIVS=0
STEM=1

BLACK=pal.RGB(0,0,0)
BLUE=pal.RGB(0,0,255)
RED=pal.RGB(255,0,0)

MOORE=pal.MooreHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Seed(grid:pal.AgentGrid):
    cell=grid.NewAgentSQ(grid.xDim//2,grid.yDim//2)
    grid[cell,DIVS]=MAX_DIVS
    grid[cell,STEM]=1


@pal.njit(cache=True)
def Divide(grid:pal.AgentGrid,cell,childLoc):
    stem=grid[cell,STEM]!=0
    divs=grid[cell,DIVS]

    stemChild=False
    divsChild=divs

    if stem and pal.Random()<STEM_DIV_PROB:
        stemChild=True
    else:
        divsChild-=1

    child=grid.NewAgentSQ(childLoc)
    grid[child,DIVS]=divsChild
    grid[child,STEM]=stemChild

    if not stem:
        grid[cell,DIVS]-=1


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,empty:pal.IList):
    stemCt=0
    nonStemCt=0

    for cell in grid.All(shuffle=True):
        if grid[cell,STEM]!=0:
            stemCt+=1
        else:
            nonStemCt+=1

        if pal.Random()<DEATH_PROB:
            grid.Dispose(cell)
            continue

        if pal.Random()<DIV_PROB:
            empty.Clear()

            for x,y in grid.Hood(MOORE,grid.XSQ(cell),grid.YSQ(cell),unroll=True):
                if grid.counts[x,y]==0:
                    empty.Append(grid.ToI(x,y))

            if len(empty):
                if grid[cell,DIVS]==0:
                    grid.Dispose(cell)
                    continue

                Divide(grid,cell,empty.Random())

    return stemCt,nonStemCt


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,pix:pal.Pix):
    pix[:]=BLACK

    for cell in grid.All():
        if grid[cell,STEM]!=0:
            pix[grid.I(cell)]=RED
        else:
            pix[grid.I(cell)]=pal.ColorScale((BLACK,BLUE),(grid[cell,DIVS]+1.0)/MAX_DIVS)


def main():
    # pal.FastMode()
    pal.Seed(0)

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Stem Cell CA")

    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=2,isStackable=False)
    empty=pal.NewIList()

    pal.AwaitWindows()

    for tick in range(RUN_TICKS):
        if grid.GetPop()==0:
            Seed(grid)

        stemCt,nonStemCt=Step(grid,empty)

        Draw(grid,pix)
        win.Update()

        print(f"Timestep {tick}  Population {grid.GetPop()}  Stem {stemCt}  Non-stem {nonStemCt}")


if __name__=="__main__":
    main()