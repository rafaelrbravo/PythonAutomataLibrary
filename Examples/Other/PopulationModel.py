import PythonAutomataLibrary as pal

N_POPS=10
TIMESTEPS=100000
CARRYING_CAPACITY=10000


@pal.njit(cache=True)
def Step(pops:pal.PopGrid,rng:pal.Multinomial):
    totalPop=pops.GetPop()

    for i in range(N_POPS-1,-1,-1):
        n=pops[i]

        n-=rng.Binomial(n,0.01)

        nDivs=rng.Binomial(n,0.5*0.9**i*(1-totalPop/CARRYING_CAPACITY))
        n+=nDivs

        if i!=N_POPS-1:
            nMuts=rng.Binomial(nDivs,0.1)
            n-=nMuts
            pops[i+1]=pops[i+1]+nMuts

        pops[i]=n


def main():
    # pal.FastMode()

    pops=pal.NewPopGrid(dimensions=(N_POPS,))
    rng=pal.NewMultinomial()

    pops[0]=100

    for _ in range(TIMESTEPS):
        Step(pops,rng)
        print([pops[i] for i in range(len(pops))],pops.GetPop())


if __name__=="__main__":
    main()