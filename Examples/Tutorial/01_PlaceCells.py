import PythonAutomataLibrary as pal

SIZE = 30

def Setup():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
    return cells

def main():
    cells = Setup()
    print("Initial cells:", len(cells.All()))

if __name__ == "__main__":
    main()
