import PythonAutomataLibrary as pal

X=10
Y=10


@pal.njit(cache=True)
def Run(grid:pal.PDEgrid):
    for y in range(grid.yDim):
        setVal=y/grid.yDim

        for x in range(grid.xDim):
            grid[x,y]=setVal

    yGradient=(grid[5,6]-grid[5,4])/2
    xGradient=(grid[6,5]-grid[4,5])/2

    print("y gradient:",yGradient)
    print("x gradient:",xGradient)


def main():
    # pal.FastMode()

    grid=pal.NewPDEgrid(dimensions=(X,Y))
    Run(grid)


if __name__=="__main__":
    main()