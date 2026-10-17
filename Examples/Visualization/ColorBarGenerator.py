import PythonAutomataLibrary as pal

X=20
Y=200
SCALE=3

BLACK=pal.RGB(0,0,0)
BLUE=pal.RGB(0,0,255)
CYAN=pal.RGB(0,255,255)
GREEN=pal.RGB(0,255,0)
YELLOW=pal.RGB(255,255,0)
RED=pal.RGB(255,0,0)
WHITE=pal.RGB(255,255,255)

COLORS=(BLUE,CYAN,GREEN,YELLOW,RED)


@pal.njit(cache=True)
def Draw(pix:pal.Pix):
    for x in range(X):
        for y in range(Y):
            pix[x,y]=pal.ColorScale(COLORS,y/(Y-1))


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Color Scale")

    Draw(pix)
    win.Update()

    while win.IsOpen():
        pass


if __name__=="__main__":
    main()