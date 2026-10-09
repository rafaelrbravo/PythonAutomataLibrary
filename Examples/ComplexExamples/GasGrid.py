import PythonAutomataLibrary as pal
import numpy as np

X=100
Y=100
SCALE=5
INIT_POP=10000
TIMESTEPS=100000
MOVE_RAD=0.5

BLACK=pal.RGB(0,0,0)
WHITE=pal.RGB(255,255,255)
COLORS=(BLACK,WHITE)


@pal.njit(cache=True)
def Setup(grid:pal.AgentGrid):
    for _ in range(INIT_POP):
        grid.NewAgent(X/2,Y/2)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid):
    for agent in grid.All():
        angle=pal.Random()*2*np.pi
        dist=np.sqrt(pal.Random())*MOVE_RAD

        x=grid.X(agent)+np.cos(angle)*dist
        y=grid.Y(agent)+np.sin(angle)*dist

        if x<0 or x>=X:
            x=grid.X(agent)

        if y<0 or y>=Y:
            y=grid.Y(agent)

        grid.Move(agent,x,y)


@pal.njit(cache=True)
def Draw(grid:pal.AgentGrid,pix:pal.Pix):
    for i in range(len(grid)):
        pix[i]=pal.ColorScale(COLORS,min(grid.counts[i]/4,1))


def main():
    # pal.FastMode()

    grid=pal.NewAgentGrid(dimensions=(X,Y),isStackable=True)
    pix,win=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE,title="Density")

    Setup(grid)

    pal.AwaitWindows()
    for tick in range(TIMESTEPS):
        if not win.IsOpen():
            break

        Step(grid)
        Draw(grid,pix)
        win.Update()

        if tick%10000==0:
            win.Save(f"gas{tick}.png")


if __name__=="__main__":
    main()