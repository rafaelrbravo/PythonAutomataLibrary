import PythonAutomataLibrary as pal
import numpy as np

X=200
Y=200
SCALE=3
INIT_POP=500

COW_RADIUS=0.5
FORCE_EXPONENT=2
FORCE_SCALER=0.7
FRICTION=0.5
GROW_RATE=0.01
EAT_RATE=1.0
CHEMOTAX_RATE=0.1
RANDOM_MOVE_RATE=0.1

BLACK=pal.RGB(0,0,0)
GREEN=pal.RGB(0,128,0)
WHITE=pal.RGB(255,255,255)
GRASS_COLORS=(BLACK,GREEN)

X_VEL=0
Y_VEL=1


@pal.njit(cache=True)
def GradientX(grass:pal.PDEgrid,x,y):
    if x==0:
        return grass[x+1,y]-grass[x,y]
    if x==X-1:
        return grass[x,y]-grass[x-1,y]
    return (grass[x+1,y]-grass[x-1,y])/2


@pal.njit(cache=True)
def GradientY(grass:pal.PDEgrid,x,y):
    if y==0:
        return grass[x,y+1]-grass[x,y]
    if y==Y-1:
        return grass[x,y]-grass[x,y-1]
    return (grass[x,y+1]-grass[x,y-1])/2


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for _ in range(INIT_POP):
        grid.NewAgentSQ(pal.RandInt(len(grid)))


@pal.njit(cache=True)
def StepCows(grid:pal.AgentGrid,grass:pal.PDEgrid):
    for cow in grid.All():
        x=grid.X(cow)
        y=grid.Y(cow)
        xSq=grid.XSQ(cow)
        ySq=grid.YSQ(cow)

        if grass[xSq,ySq]>0:
            grass.Add(-EAT_RATE,xSq,ySq)

        xVel=grid[cow,X_VEL]
        yVel=grid[cow,Y_VEL]

        gradX=GradientX(grass,xSq,ySq)
        gradY=GradientY(grass,xSq,ySq)
        norm=np.sqrt(gradX*gradX+gradY*gradY)

        if norm>0:
            xVel+=gradX/norm*CHEMOTAX_RATE
            yVel+=gradY/norm*CHEMOTAX_RATE

        angle=pal.Random()*2*np.pi
        dist=np.sqrt(pal.Random())*RANDOM_MOVE_RATE
        xVel+=np.cos(angle)*dist
        yVel+=np.sin(angle)*dist

        for other,dx,dy,distSq in grid.AgentsInRadius(COW_RADIUS*2,x,y,exclude=cow):
            if distSq>0:
                dist=np.sqrt(distSq)
                overlap=COW_RADIUS*2-dist
                force=(FORCE_SCALER*overlap)**FORCE_EXPONENT

                xVel-=dx/dist*force
                yVel-=dy/dist*force

        newX=x+xVel
        newY=y+yVel

        if newX<0 or newX>=X:
            newX=x

        if newY<0 or newY>=Y:
            newY=y

        grid.Move(cow,newX,newY)
        grid[cow,X_VEL]=xVel*FRICTION
        grid[cow,Y_VEL]=yVel*FRICTION

    grass.Update()


@pal.njit(cache=True)
def GrowGrass(grass:pal.PDEgrid):
    for i in range(len(grass)):
        grass.Add(GROW_RATE,i)

    grass.Update()

    for i in range(len(grass)):
        if grass[i]>1:
            grass[i]=1


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,grass:pal.PDEgrid,draw:pal.OpenGLDraw):
    draw.Clear()

    for i in range(len(grass)):
        x=grass.ItoX(i)
        y=grass.ItoY(i)
        draw.BoxSQ(pal.ColorScale(GRASS_COLORS,grass[i]),x,y)

    for cow in grid.All():
        draw.Circle(COW_RADIUS,WHITE,grid.X(cow),grid.Y(cow))


def main():
    pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,title="Chemotaxis")
    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=2,isStackable=True)
    grass=pal.NewPDEgrid(dimensions=(X,Y))

    grass[:]=1.0
    Setup(grid)

    pal.AwaitWindows()
    while win.IsOpen():
        StepCows(grid,grass)
        GrowGrass(grass)
        Draw(grid,grass,draw)
        win.Update()


if __name__=="__main__":
    main()