import PythonAutomataLibrary as pal

X=20
Y=10
Z=10
SCALE=32

GRAY=pal.RGB(128,128,128)
WHITE=pal.RGB(255,255,255)
RED=pal.RGB(255,0,0)
GREEN=pal.RGB(0,255,0)
BLUE=pal.RGB(0,0,255)


@pal.njit(cache=True)
def Draw(draw:pal.OpenGLDraw):
    draw.Clear()
    draw.Circle(1,RED,20,0,0)
    draw.Circle(1,GREEN,0,30,0)
    draw.Circle(1,BLUE,0,0,10)


def main():
    # pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,zDim=Z,scale=SCALE,title="testing")
    win.Background(GRAY)
    win.Borders(1,WHITE)

    pal.AwaitWindows()
    while win.IsOpen():
        Draw(draw)
        win.Update()


if __name__=="__main__":
    main()