import PythonAutomataLibrary as pal

RULE=110
DOMAIN_SIZE=200
X=DOMAIN_SIZE*2-1
Y=DOMAIN_SIZE
SCALE=3

BLACK=pal.RGB(0,0,0)
WHITE=pal.RGB(255,255,255)

NEIGHBORHOOD=((-1,),(0,),(1,))


@pal.njit(cache=True)
def Step(grid:pal.Grid,y):
    for x in range(1,X-1):
        state=0

        for nx in grid.Hood(NEIGHBORHOOD,x,unroll=True):
            state=(state<<1)|grid[nx,y+1]

        grid[x,y]=(RULE>>state)&1


@pal.njit(cache=True)
def Draw(grid:pal.Grid,pix:pal.Pix):
    for i in range(len(grid)):
        pix[i]=BLACK if grid[i] else WHITE


def main():
    # pal.FastMode()

    grid=pal.NewGrid(dimensions=(X,Y),dtype=bool)
    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title=f"Rule {RULE}")

    grid[X//2,Y-1]=True

    pal.AwaitWindows()
    Draw(grid,pix)
    win.Update()

    win.StartGif(f"rule{RULE}.gif",delay=10)
    win.AddGifFrame()

    for y in range(Y-2,-1,-1):
        Step(grid,y)
        Draw(grid,pix)
        win.Update()
        win.AddGifFrame()

    win.StopGif()

    while win.IsOpen():
        pass


if __name__=="__main__":
    main()