import PythonAutomataLibrary as pal

X=100
Y=100
SCALE=5

C1=pal.RGB(0,114,189)
C2=pal.RGB(217,83,25)
C3=pal.RGB(237,177,32)
C4=pal.RGB(126,47,142)


@pal.njit(cache=True)
def Draw(pix:pal.Pix):
    for x in range(X):
        for y in range(Y):
            bottom=pal.ColorScale((C2,C4),x/X)
            top=pal.ColorScale((C1,C3),x/X)
            pix[x,y]=pal.ColorScale((bottom,top),y/Y)


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE)

    Draw(pix)
    win.Update()
    win.Save("test.png")

    while win.IsOpen():
        pass


if __name__=="__main__":
    main()