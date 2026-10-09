import time, numpy as np
from numba import njit
import NativeCore as pal
MOORE=np.array([-1,-1,-1,0,-1,1,0,-1,0,1,1,-1,1,0,1,1],np.int32)
g=pal.NewAgentGrid((100,100),1,False,False);q=pal.AgentQueryList(g,MOORE)
@njit
def setup(g):
    for x in range(40,61):
        for y in range(40,61):
            if (x-50)**2+(y-50)**2<=100:
                a=g.NewAgentSQ(x,y);g.SetP(a,0,1)
@njit
def step(g,q):
    aa=g.All(True)
    for a in aa:
        if np.random.random()<.01:g.Dispose(a)
        elif np.random.random()<.2:
            g.MapEmptyHood(q,g.XSQ(a),g.YSQ(a))
            if len(q):
                c=g.NewAgentI(q.Random());g.SetP(c,0,g.GetP(a,0))
t=time.perf_counter();setup(g);print('setup',time.perf_counter()-t,g.GetPop())
t=time.perf_counter();step(g,q);print('step',time.perf_counter()-t,g.GetPop())
@njit
def combo(pg,pd):
    pg.SetI(10,0);pg.AddI(2,0);pg.Update();pd.SetI(1.,0);pd.Diffusion(.1);pd.Update();return pg.GetPop(),pd.GetI(0)
pg=pal.NewPopGrid((20,20),1000,False);pd=pal.NewPDEgrid((20,20),False)
t=time.perf_counter();print('combo',combo(pg,pd),'compile',time.perf_counter()-t)
