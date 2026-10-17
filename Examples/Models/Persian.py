import time
import PythonAutomataLibrary as pal

DIM=101
SCALE=5
TIMESTEPS=200
PAUSE=50

B=1.61

COOPERATOR=0
DEFECTOR=1

RED=pal.RGB(255,0,0)
GREEN=pal.RGB(0,255,0)
YELLOW=pal.RGB(255,255,0)
BLUE=pal.RGB(0,0,255)

MOORE=pal.MooreHood(2,excludeCenter=True)


@pal.njit(cache=True)
def Setup(types:pal.Grid):
    types[:]=COOPERATOR
    types[DIM//2,DIM//2]=DEFECTOR


@pal.njit(cache=True)
def DetermineFitness(types:pal.Grid,fitness:pal.Grid):
    for x in range(DIM):
        for y in range(DIM):
            cooperators=0
            defectors=0

            for nx,ny in types.Hood(MOORE,x,y,unroll=True):
                if types[nx,ny]==COOPERATOR:
                    cooperators+=1
                else:
                    defectors+=1

            if types[x,y]==COOPERATOR:
                fitness[x,y]=cooperators
            else:
                fitness[x,y]=B*cooperators


@pal.njit(cache=True)
def SetFutureTypes(types:pal.Grid,futureTypes:pal.Grid,fitness:pal.Grid):
    for x in range(DIM):
        for y in range(DIM):
            maxFitness=0.0
            futureType=types[x,y]

            for nx,ny in types.Hood(MOORE,x,y,unroll=True):
                if fitness[nx,ny]>maxFitness:
                    maxFitness=fitness[nx,ny]
                    futureType=types[nx,ny]

            futureTypes[x,y]=futureType


@pal.njit(cache=True)
def Draw(types:pal.Grid,futureTypes:pal.Grid,pix:pal.Pix):
    for i in range(len(types)):
        oldType=types[i]
        newType=futureTypes[i]

        if oldType==DEFECTOR and newType==DEFECTOR:
            pix[i]=RED
        elif oldType==DEFECTOR and newType==COOPERATOR:
            pix[i]=GREEN
        elif oldType==COOPERATOR and newType==DEFECTOR:
            pix[i]=YELLOW
        else:
            pix[i]=BLUE


@pal.njit(cache=True)
def Update(types:pal.Grid,futureTypes:pal.Grid):
    for i in range(len(types)):
        types[i]=futureTypes[i]


@pal.njit(cache=True)
def Step(types:pal.Grid,futureTypes:pal.Grid,fitness:pal.Grid,pix:pal.Pix):
    DetermineFitness(types,fitness)
    SetFutureTypes(types,futureTypes,fitness)
    Draw(types,futureTypes,pix)
    Update(types,futureTypes)


def main():
    # pal.FastMode()

    types=pal.NewGrid(dimensions=(-DIM,-DIM),dtype="int8")
    futureTypes=pal.NewGrid(dimensions=(-DIM,-DIM),dtype="int8")
    fitness=pal.NewGrid(dimensions=(-DIM,-DIM),dtype="float64")
    pix,win=pal.StartPixWindow(xDim=DIM,yDim=DIM,scale=SCALE,title="Persian Carpet")

    Setup(types)
    futureTypes[:]=types[:]

    Draw(types,futureTypes,pix)
    win.Update()

    win.StartGif("persian_carpet.gif",delay=100)

    pal.AwaitWindows()
    for _ in range(TIMESTEPS):
        if not win.IsOpen():
            break

        Step(types,futureTypes,fitness,pix)
        win.Update()
        win.AddGifFrame()

        if PAUSE>0:
            time.sleep(PAUSE/1000)

    win.StopGif()


if __name__=="__main__":
    main()