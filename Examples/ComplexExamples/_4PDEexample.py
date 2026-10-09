import math
import numpy as np
import PythonAutomataLibrary as pal
import time

X=400
Y=400
SCALE=2

N_SINKS=100
SINK_DIST=10
TICK_PAUSE=100

EMPTY=0
SRC=1
SINK=2

BLACK=pal.RGB(0,0,0)
GREEN=pal.RGB(0,255,0)
CYAN=pal.RGB(0,255,255)

HEAT_COLORS=(
    pal.RGB(0,0,0),
    pal.RGB(0,0,255),
    pal.RGB(0,255,255),
    pal.RGB(0,255,0),
    pal.RGB(255,255,0),
    pal.RGB(255,0,0),
)


@pal.njit(cache=True)
def Setup(cells:pal.Grid):
    centerX=cells.xDim//2
    centerY=cells.yDim//2

    for x in range(centerX,centerX+3):
        for y in range(centerY,centerY+3):
            cells[x,y]=SRC

    sinksPlaced=0

    while sinksPlaced<N_SINKS:
        i=pal.RandInt(len(cells))

        if cells[i]!=EMPTY:
            continue

        dx=cells.ItoX(i)-centerX
        dy=cells.ItoY(i)-centerY

        if dx*dx+dy*dy>SINK_DIST*SINK_DIST:
            cells[i]=SINK
            sinksPlaced+=1


@pal.njit(cache=True)
def Step(cells:pal.Grid,diff:pal.PDEgrid,stepI):
    advectionX=math.sin(stepI/1000.0)*0.2
    advectionY=math.cos(stepI/1000.0)*0.2

    for i in range(len(cells)):
        if cells[i]==SRC:
            diff[i]=1.0
        elif cells[i]==SINK:
            diff[i]=0.0

    diff.Advection(advectionX,advectionY,xMinBC=0,yMinBC=0,xMaxBC=0,yMaxBC=0)
    diff.Diffusion((math.sin(stepI/250.0)+1.0)*0.05,xMinBC=0,yMinBC=0,xMaxBC=0,yMaxBC=0)
    diff.Update()


@pal.njit(cache=True)
def DrawCells(cells:pal.Grid,pix:pal.Pix):
    pix[:]=BLACK

    for i in range(len(cells)):
        if cells[i]==SRC:
            pix[i]=GREEN
        elif cells[i]==SINK:
            pix[i]=CYAN


@pal.njit(cache=True)
def DrawDiff(diff:pal.PDEgrid,pix:pal.Pix):
    for i in range(len(diff)):
        pix[i]=pal.ColorScale(HEAT_COLORS,diff[i]*4.0)


def main():
    # pal.FastMode()
    pal.Seed(0)

    cellPix,cellWin=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Sources and Sinks")
    diffPix,diffWin=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Diffusion")

    cells=pal.NewGrid(dimensions=(X,Y),dtype=np.int8)
    diff=pal.NewPDEgrid(dimensions=(X,Y))

    Setup(cells)

    pal.AwaitWindows()

    stepI=0

    while cellWin.IsOpen() and diffWin.IsOpen():

        Step(cells,diff,stepI)
        DrawCells(cells,cellPix)
        DrawDiff(diff,diffPix)

        cellWin.Update()
        diffWin.Update()

        stepI+=1


if __name__=="__main__":
    main()