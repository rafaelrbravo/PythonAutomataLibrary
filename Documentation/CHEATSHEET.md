# PAL Cheatsheet

`import PythonAutomataLibrary as pal`

**Default:** safe mode. Call `pal.FastMode()` **before** constructing any PAL object to opt into fast mode. Compile model functions with `@pal.njit`.

## Create state

| Need | Constructor |
| --- | --- |
| Typed lattice | `pal.NewGrid((nx, ny), dtype)` |
| Individual agents | `pal.NewAgentGrid((nx, ny), numAgentProps=0, isStackable=False)` |
| Population counts | `pal.NewPopGrid((nx, ny), capacity=None)` |
| Continuous field | `pal.NewPDEgrid((nx, ny))` |
| Integer list | `pal.NewIList()` |
| Random count sampler | `pal.NewMultinomial()` |

Dimensions: 1–3 axes; negative size wraps that axis (`(-nx, ny)`). `pal.NewAgentGrid(())` supports nonspatial populations.

## Shared geometry

| Task | Syntax |
| --- | --- |
| Size / wrap | `grid.xDim`, `grid.yDim`, `grid.nDims`, `grid.wrapX` |
| Coordinates → index | `grid.ToI(x, y)` |
| Index → coordinates | `grid.ItoX(i)`, `grid.ItoY(i)` |
| Rectangular region | `grid.Box(x1, x2, y1, y2)` (half-open) |
| Neighbor sites | `grid.Hood(hood, x, y)` |
| Neighborhood offsets | `pal.MooreHood(2, True)`, `pal.VonNeumannHood(2)`, `pal.CircleHood(2, 3)` |

## Grid

```python
g = pal.NewGrid((40, 40), float)
x, y = 10, 12
g[x, y] = 1.0
value = g[x, y]
copy = g[2:5, 3:8]    # detached NumPy array
```

## AgentGrid

```python
agents = pal.NewAgentGrid((40, 40), numAgentProps=2)
a = agents.NewAgentSQ(10, 12)
agents[a, 0] = 1.5            # property
agents.MoveSQ(a, 11, 12)
for a in agents.All():        # snapshot; safe to Dispose during loop
    agents.Dispose(a)
```

`agents.GetPop()`, `agents.Alive(a)`, `agents.AgentsAt(x, y)`, `agents.LastAgent(x, y)`, `agents.counts[x, y]`. Continuous positions: `NewAgent(x, y)`, `Move(a, x, y)`, `X(a)`, `Y(a)`.

## PopGrid and PDEgrid

```python
pop = pal.NewPopGrid((40, 40))
pop.Add(1, x, y)  # pending
pop.Update()      # apply simultaneously
pop.GetPop()      # total
pop.Reset()       # clear current + pending

field = pal.NewPDEgrid((40, 40))
field.SetTimeSpaceStep(dt, dx, dy)
field.Diffusion(rateConstant)
field.Update()
```

Direct `pop[x, y] = n` / `field[x, y] = value` changes current state immediately. `Add` changes only pending state until `Update`.

PDE methods: `Diffusion`, `DiffusionMask`, `DiffusionField`, `DiffusionInterfaces`, `DiffusionADI`, `DiffusionRadialCircle`, `DiffusionRadialSphere`, `Advection`, `AdvectionField`, `AdvectionInterfaces`.

## Lists and randomness

```python
items = pal.NewIList()
items.Append(4).Append(8)
for i in items.Iter(): pass  # live/no copy
snapshot = items.All()       # detached copy
items.Shuffle()
pal.Seed(123)
u = pal.Random()
i = pal.RandInt(10)          # 0..9
m = pal.NewMultinomial()
count = m.Binomial(20, 0.3)
```

## Draw

```python
pix, window = pal.StartPixWindow(40, 40, scale=4)
pix[x, y] = 0xFF0000
window.Update()
window.Save("frame.png", block=True)
window.Close()
```

For 2D/3D geometry use `pal.StartOpenGLWindow(...)`; for unattended output pass `headless=True` with a supported backend. Use `window.StartGif(path)`, `AddGifFrame()`, and `StopGif()` for animation.

**More detail:** [Manual](MANUAL.md) · [API Guide](API_GUIDE.md) · [API Reference](API_REFERENCE.md).
