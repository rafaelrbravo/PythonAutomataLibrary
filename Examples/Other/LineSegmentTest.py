import PythonAutomataLibrary as pal

X=10
Y=10
Z=10
SCALE=50

RED=pal.RGB(255,0,0)


@pal.njit(cache=True)
def Draw(draw:pal.OpenGLDraw):
    draw.Clear()
    draw.Line(1,RED,0,0,10,10,0,10)


def main():
    # pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,zDim=Z,scale=SCALE,title="segment")

    pal.AwaitWindows()
    while win.IsOpen():
        Draw(draw)
        win.Update()


if __name__=="__main__":
    main()