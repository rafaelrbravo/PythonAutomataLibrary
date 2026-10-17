import PythonAutomataLibrary as pal

X=50
Y=50
SCALE=10
MOVE_PROB=0.1
STARTING_POP=100000000

COLORSCALE=(
    pal.RGB(0,0,0),
    pal.RGB(255,0,0),
    pal.RGB(255,255,0),
    pal.RGB(255,255,255)
)

VN=pal.VonNeumannHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Step(cells:pal.PopGrid,mn:pal.Multinomial):
    cells.Update()

    # Move cells.
    for i in cells.All():
        mn.Setup(cells[i])

        for x,y in cells.Hood(VN,cells.ItoX(i),cells.ItoY(i),unroll=True):
            movePop=mn.Sample(MOVE_PROB)
            cells.Add(-movePop,i)
            cells.Add(movePop,x,y)


@pal.njit(cache=True)
def Draw(cells:pal.PopGrid,pix:pal.Pix):
    for i in range(len(cells)):
        density=min(cells[i]/100000.0,1.0)
        pix[cells.ItoX(i),cells.ItoY(i)]=pal.ColorScale(COLORSCALE,density)


def main():
    pal.FastMode()

    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE)

    cells=pal.NewPopGrid(dimensions=(X,Y))
    mn=pal.NewMultinomial()

    cells[0,0]=STARTING_POP

    pal.AwaitWindows()
    while win.IsOpen():
        Step(cells,mn)
        Draw(cells,pix)
        win.Update()


if __name__=="__main__":
    main()