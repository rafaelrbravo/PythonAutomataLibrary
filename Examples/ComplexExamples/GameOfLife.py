import time
import PythonAutomataLibrary as pal

X=1000
Y=1000
SCALE=1
LIVING_PROB=0.35
TIMESTEPS=10000000
REFRESH_RATE=0

LIVE=pal.RGB(255,0,0)
DEAD=pal.RGB(0,0,0)

MOORE=pal.MooreHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Setup(grid:pal.Grid):
    for i in range(len(grid)):
        grid[i]=pal.Random()<LIVING_PROB


@pal.njit(cache=True)
def Step(grid:pal.Grid,nextGrid:pal.Grid):
    liveCt=0

    for x in range(X):
        for y in range(Y):
            neighbors=0

            for nx,ny in grid.Hood(MOORE,x,y,unroll=True):
                if grid[nx,ny]:
                    neighbors+=1

            alive=grid[x,y]
            nextAlive=(alive and (neighbors==2 or neighbors==3)) or (not alive and neighbors==3)
            nextGrid[x,y]=nextAlive

            if nextAlive:
                liveCt+=1

    return liveCt


@pal.njit(cache=True)
def Copy(nextGrid:pal.Grid,grid:pal.Grid):
    for i in range(len(grid)):
        grid[i]=nextGrid[i]


@pal.njit(cache=True)
def Draw(grid:pal.Grid,pix:pal.Pix):
    for i in range(len(grid)):
        pix[i]=LIVE if grid[i] else DEAD


def main():
    # pal.FastMode()

    grid=pal.NewGrid(dimensions=(-X,-Y),dtype=bool)
    nextGrid=pal.NewGrid(dimensions=(-X,-Y),dtype=bool)
    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Game of Life")

    Setup(grid)

    pal.AwaitWindows()
    for tick in range(TIMESTEPS):
        if not win.IsOpen():
            break

        liveCt=Step(grid,nextGrid)
        Copy(nextGrid,grid)

        Draw(grid,pix)
        win.Update()

        print(f"Tick {tick+1}, Population {liveCt}")

        if REFRESH_RATE>0:
            time.sleep(REFRESH_RATE/1000)


if __name__=="__main__":
    main()