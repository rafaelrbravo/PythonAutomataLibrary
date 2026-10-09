import time
import numpy as np
import PythonAutomataLibrary as pal

X=100
Y=100
SCALE=5
TIMESTEPS=1000
DEATH_PROB=0.1
BIRTH_PROB=0.2

COLOR=0


@pal.njit(cache=True)
def NewCell(grid:pal.AgentGrid,x,y):
    agent=grid.NewAgent(x,y)
    color=pal.RGB(pal.RandInt(256),pal.RandInt(256),pal.RandInt(256))
    grid[agent,COLOR]=color
    return agent


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid):
    # Snapshot + shuffle: newborn agents wait until the next timestep.
    for agent in grid.All(shuffle=True):
        if pal.Random()<DEATH_PROB:
            grid.Dispose(agent)
            continue

        if pal.Random()<BIRTH_PROB and grid.counts[grid.I(agent)]<5:
            NewCell(grid,grid.X(agent),grid.Y(agent))

        # Uniform random point within a radius-0.5 circle.
        angle=pal.Random()*2*np.pi
        rad=np.sqrt(pal.Random())*0.5

        x=grid.X(agent)
        y=grid.Y(agent)
        newX=x+np.cos(angle)*rad
        newY=y+np.sin(angle)*rad

        # Independently cancel movement along an axis that would leave
        # the non-wrapped domain.
        if newX<0 or newX>=grid.xDim:
            newX=x
        if newY<0 or newY>=grid.yDim:
            newY=y

        grid.Move(agent,newX,newY)


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,draw:pal.OpenGLDraw):
    draw.Clear()
    for agent in grid.All():
        draw.Circle(0.5,np.uint32(grid[agent,COLOR]),grid.X(agent),grid.Y(agent))


def main():
    pal.FastMode()

    draw,win=pal.StartOpenGLWindow(xDim=X,yDim=Y,title="BirthDeathOffLattice")
    grid=pal.NewAgentGrid(dimensions=(X,Y),numAgentProps=1,isStackable=True)

    pal.AwaitWindows()
    for _ in range(TIMESTEPS):
        if not win.IsOpen():
            break

        if grid.GetPop()==0:
            NewCell(grid,X/2,Y/2)

        Step(grid)
        Draw(grid,draw)
        win.Update()

        time.sleep(0.01)


if __name__=="__main__":
    main()