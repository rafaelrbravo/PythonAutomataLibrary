import PythonAutomataLibrary as pal


def main():
    # pal.FastMode()
    mn=pal.NewMultinomial()
    print(mn.Binomial(2**63-1,0.1)/(2**63-1))


if __name__=="__main__":
    main()