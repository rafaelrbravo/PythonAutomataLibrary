import PythonAutomataLibrary as pal
import numpy as np

X=30
Y=30
SCALE=30
TIMESTEPS=1000

INIT_POP=50
INIT_RADIUS=5
PROP_PURPLE=0.5

RADIUS=0.25
FORCE_SCALER=0.25
FRICTION=0.5

PURPLE_DIV_BIAS=0.01
PINK_DIV_BIAS=0.02
PURPLE_INHIB_WEIGHT=0.02
PINK_INHIB_WEIGHT=0.05

WHITE=pal.RGB(248,255,252)
PURPLE=pal.RGB(77,0,170)
PINK=pal.RGB(222,0,109)
CYTOPLASM=pal.RGB(191,156,147)

TYPE=0
X_VEL=1
Y_VEL=2
FORCE_SUM=3


@pal.njit(cache=True)
def NewCell(grid:pal.AgentGrid,x,y,cellType):
    cell=grid.NewAgent(x,y)
    grid[cell,TYPE]=cellType
    return cell


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for _ in range(INIT_POP):
        angle=pal.Random()*2*np.pi
        dist=np.sqrt(pal.Random())*INIT_RADIUS

        x=X/2+np.cos(angle)*dist
        y=Y/2+np.sin(angle)*dist
        cellType=0 if pal.Random()<PROP_PURPLE else 1

        NewCell(grid,x,y,cellType)


@pal.njit(cache=True)
def CalcForces(grid:pal.AgentGrid):
    for cell in grid.All():
        x=grid.X(cell)
        y=grid.Y(cell)

        xVel=grid[cell,X_VEL]
        yVel=grid[cell,Y_VEL]
        forceSum=0.0

        for other,dx,dy,distSq in grid.AgentsInRadius(RADIUS*2,x,y,exclude=cell):
            if distSq>0:
                dist=np.sqrt(distSq)
                overlap=RADIUS*2-dist
                force=FORCE_SCALER*overlap

                xVel-=dx/dist*force
                yVel-=dy/dist*force
                forceSum+=force

        grid[cell,X_VEL]=xVel
        grid[cell,Y_VEL]=yVel
        grid[cell,FORCE_SUM]=forceSum


@pal.njit(cache=True)
def MoveDivide(grid:pal.AgentGrid):
    for cell in grid.All():
        xVel=grid[cell,X_VEL]
        yVel=grid[cell,Y_VEL]

        grid.Move(cell,grid.InWrapX(grid.X(cell)+xVel),grid.InWrapY(grid.Y(cell)+yVel))

        grid[cell,X_VEL]=xVel*FRICTION
        grid[cell,Y_VEL]=yVel*FRICTION

        cellType=int(grid[cell,TYPE])
        forceSum=grid[cell,FORCE_SUM]

        if cellType==0:
            divProb=np.tanh(PURPLE_DIV_BIAS-forceSum*PURPLE_INHIB_WEIGHT)
        else:
            divProb=np.tanh(PINK_DIV_BIAS-forceSum*PINK_INHIB_WEIGHT)

        if pal.Random()<divProb:
            angle=pal.Random()*2*np.pi
            divRad=RADIUS*2/3

            dx=np.cos(angle)*divRad
            dy=np.sin(angle)*divRad

            x=grid.X(cell)
            y=grid.Y(cell)

            grid.Move(cell,grid.InWrapX(x-dx),grid.InWrapY(y-dy))
            NewCell(grid,grid.InWrapX(x+dx),grid.InWrapY(y+dy),cellType)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid):
    CalcForces(grid)
    MoveDivide(grid)


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,draw:pal.OpenGLDraw):
    draw.Clear()

    for cell in grid.All():
        draw.Circle(RADIUS,CYTOPLASM,grid.X(cell),grid.Y(cell))

    for cell in grid.All():
        color=PURPLE if int(grid[cell,TYPE])==0 else PINK
        draw.Circle(RADIUS/3,color,grid.X(cell),grid.Y(cell))


@pal.njit(cache=True)
def CountTypes(grid:pal.AgentGrid):
    purple=0
    pink=0

    for cell in grid.All():
        if int(grid[cell,TYPE])==0:
            purple+=1
        else:
            pink+=1

    return pink,purple


def main():
    pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,title="Off Lattice Example")
    pal.Seed(0)

    grid=pal.NewAgentGrid(dimensions=(-X,-Y),numAgentProps=4,isStackable=True)
    Setup(grid)

    win.Background(WHITE)

    pal.AwaitWindows()
    with open("PopOut.csv","w") as out:
        for _ in range(TIMESTEPS):
            if not win.IsOpen():
                break

            Step(grid)
            Draw(grid,draw)
            win.Update()

            pink,purple=CountTypes(grid)
            out.write(f"{pink},{purple}\n")


if __name__=="__main__":
    main()