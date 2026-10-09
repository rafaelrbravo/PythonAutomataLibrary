import time
import PythonAutomataLibrary as pal

X=10
Y=10
Z=10
SCALE_2D=20
SCALE_3D=50
TIMESTEPS=10000

COLORSCALE=(pal.RGB(0,0,255),pal.RGB(0,255,255),pal.RGB(0,255,0),pal.RGB(255,255,0),pal.RGB(255,0,0))

WHITE=pal.RGB(255,255,255)
BLACK=pal.RGB(0,0,0)
GREEN=pal.RGB(0,255,0)


@pal.njit(cache=True)
def Step(grid:pal.AgentGrid,agent):
    x=grid.XSQ(agent)+pal.RandInt(3)-1
    y=grid.YSQ(agent)+pal.RandInt(3)-1
    z=grid.ZSQ(agent)+pal.RandInt(3)-1

    # Match HAL MoveSafeSQ on a non-wrapped grid:
    # independently cancel movement along axes that leave the domain.
    if x<0 or x>=grid.xDim:
        x=grid.XSQ(agent)
    if y<0 or y>=grid.yDim:
        y=grid.YSQ(agent)
    if z<0 or z>=grid.zDim:
        z=grid.ZSQ(agent)

    grid.MoveSQ(agent,x,y,z)


@pal.njit(cache=True)
def Draw2D(grid:pal.AgentGrid,agent,pix:pal.Pix):
    pix[:]=BLACK
    pix[grid.XSQ(agent),grid.YSQ(agent)]=pal.ColorScale(COLORSCALE,grid.ZSQ(agent)/grid.zDim)


@pal.njit(cache=True)
def Draw3D(grid:pal.AgentGrid,agent,draw:pal.OpenGLDraw):
    draw.Clear()
    draw.Circle(0.5,GREEN,grid.X(agent),grid.Y(agent),grid.Z(agent))


def main():
    # pal.FastMode()

    pix,win2D=pal.StartPixWindow(xDim=X,yDim=Y,scale=SCALE_2D)
    draw,win3D=pal.StartOpenGLWindow(xDim=X,yDim=Y,zDim=Z,scale=SCALE_3D)
    win3D.Borders(1,WHITE)

    grid=pal.NewAgentGrid(dimensions=(X,Y,Z),numAgentProps=0,isStackable=False)
    agent=grid.NewAgentSQ(5,5,5)

    pal.AwaitWindows()
    for _ in range(TIMESTEPS):
        if not win2D.IsOpen() or not win3D.IsOpen():
            break

        Step(grid,agent)
        Draw2D(grid,agent,pix)
        Draw3D(grid,agent,draw)

        win3D.Update()
        win2D.Update()
        time.sleep(0.01)


if __name__=="__main__":
    main()