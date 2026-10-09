import time
import PythonAutomataLibrary as pal

X=10
Y=10
Z=10
SCALE=50
TIMESTEPS=10000

BLUE=pal.RGB(0,0,255)
BLACK=pal.RGB(0,0,0)
RED=pal.RGB(255,0,0)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,agent):
    # Uniform random point within a radius-0.4 sphere.
    while True:
        dx=pal.Random()*0.8-0.4
        dy=pal.Random()*0.8-0.4
        dz=pal.Random()*0.8-0.4

        if dx*dx+dy*dy+dz*dz<=0.16:
            break

    x=grid.X(agent)
    y=grid.Y(agent)
    z=grid.Z(agent)

    newX=x+dx
    newY=y+dy
    newZ=z+dz

    # Match HAL MoveSafePT behavior on a non-wrapped grid:
    # independently cancel movement along axes that leave the domain.
    if newX<0 or newX>=grid.xDim:
        newX=x
    if newY<0 or newY>=grid.yDim:
        newY=y
    if newZ<0 or newZ>=grid.zDim:
        newZ=z

    grid.Move(agent,newX,newY,newZ)


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,agent,draw:pal.OpenGLDraw):
    draw.Clear()
    draw.Circle(0.5,RED,grid.X(agent),grid.Y(agent),grid.Z(agent))


def main():
    # pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,zDim=Z,scale=SCALE,title="3D")
    win.Background(BLUE)
    win.Borders(1,BLACK)

    grid=pal.NewAgentGrid(dimensions=(X,Y,Z),numAgentProps=0,isStackable=False)
    agent=grid.NewAgentSQ(5,5,5)

    pal.AwaitWindows()
    for _ in range(TIMESTEPS):
        if not win.IsOpen():
            break

        Step(grid,agent)
        Draw(grid,agent,draw)
        win.Update()
        time.sleep(0.01)


if __name__=="__main__":
    main()