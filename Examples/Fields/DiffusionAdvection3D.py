import time
import PythonAutomataLibrary as pal

X=20
Y=20
Z=20
SCALE=10

COLORS=(pal.RGB(0,0,0),pal.RGB(0,0,255),pal.RGB(0,255,255),pal.RGB(255,255,255))


@pal.njit(cache=True)
def Step(grid:pal.PDEgrid):
    grid.Advection(vx=0.1,vy=0,vz=0.1)
    grid.Diffusion(0.13)
    grid.Update()


@pal.njit(cache=True)
def Draw(grid:pal.PDEgrid,pix:pal.Pix):
    for x in range(X):
        for y in range(Y):
            value=0.0
            for z in range(Z):
                value+=grid[x,y,z]
            pix[x,y]=pal.ColorScale(COLORS,min(value*1000,1))


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE)
    grid=pal.NewPDEgrid(dimensions=(-X,-Y,-Z))

    grid[X//2,Y//2,Z//2]=1

    pal.AwaitWindows()
    while win.IsOpen():
        Step(grid)
        Draw(grid,pix)
        win.Update()
        time.sleep(0.1)


if __name__=="__main__":
    main()