import PythonAutomataLibrary as pal

X=100
Y=6
SCALE=10

R=pal.RGB(255,0,0)
G=pal.RGB(0,255,0)
B=pal.RGB(0,0,255)

COLORS=(
    (B,G,R),
    (B,R,G),
    (G,B,R),
    (G,R,B),
    (R,B,G),
    (R,G,B)
)


@pal.njit(cache=True)
def Draw(pix:pal.Pix):
    for y in range(Y):
        for x in range(X):
            pix[x,Y-1-y]=pal.ColorScale(COLORS[y],x/X)


def main():
    # pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Colors")

    Draw(pix)
    win.Update()

    while win.IsOpen():
        pass


if __name__=="__main__":
    main()