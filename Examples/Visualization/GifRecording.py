import PythonAutomataLibrary as pal

X=10
Y=10
SCALE=100


@pal.njit(cache=True)
def Draw(pix:pal.Pix):
    pix[4,4]=pal.RGB(
        int(pal.Random()*256),
        int(pal.Random()*256),
        int(pal.Random()*256)
    )


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="test")

    Draw(pix)
    win.Update()
    win.Save("test.jpg")

    win.StartGif("test.gif",delay=100)
    pal.AwaitWindows()
    for _ in range(10):
        Draw(pix)
        win.Update()
        win.AddGifFrame()
    win.StopGif()


if __name__=="__main__":
    main()