import PythonAutomataLibrary as pal

SIZE = 30

def Setup(cells):
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells.NewAgentSQ(x, y)
def main():
    cells = pal.NewAgentGrid((SIZE, SIZE), isStackable=False)
    Setup(cells)
    print("Initial cells:", len(cells.All()))

if __name__ == "__main__":
    main()
