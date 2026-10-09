import time
import PythonAutomataLibrary as pal

X=20
Y=10
TIME=10000
SCALE=10

COLORS=(pal.RGB(0,0,0),pal.RGB(255,0,0),pal.RGB(255,255,0),pal.RGB(255,255,255))


@pal.njit(cache=True)
def Step(g1:pal.PDEgrid,g2:pal.PDEgrid):
    for y in range(3,Y-3):
        g1[10,y]=1
        g2[10,y]=1

    g1.Advection(vx=0.01,vy=0.01,xMaxBC=0,yMaxBC=0)
    g1.Update()

    g2.DiffusionADI(0.01,xMaxBC=0,xMinBC=0)
    g2.Update()


@pal.njit(cache=True)
def Draw(g1:pal.PDEgrid,g2:pal.PDEgrid,pix1:pal.Pix,pix2:pal.Pix):
    for x in range(X):
        for y in range(Y):
            pix1[x,y]=pal.ColorScale(COLORS,g1[x,y])
            pix2[x,y]=pal.ColorScale(COLORS,g2[x,y])


def main():
    # pal.FastMode()

    pix1,win1=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="advection")
    pix2,win2=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="diffusion")

    g1=pal.NewPDEgrid(dimensions=(X,Y))
    g2=pal.NewPDEgrid(dimensions=(X,Y))

    pal.AwaitWindows()
    for _ in range(TIME):
        if not win1.IsOpen() or not win2.IsOpen():
            break

        Step(g1,g2)
        Draw(g1,g2,pix1,pix2)
        win1.Update()
        win2.Update()
        time.sleep(0.005)


if __name__=="__main__":
    main()