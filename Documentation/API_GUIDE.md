# PAL API Guide

This guide is the compact human reference for PAL's public API. It consolidates behavior shared across PAL objects instead of repeating the same geometry, indexing, wrapping, and compilation rules for each class. For current public signatures, use the generated [API Reference](API_REFERENCE.md). For learning PAL from the beginning, use the [Manual](MANUAL.md). For runnable syntax at a glance, use the [Cheatsheet](CHEATSHEET.md).

## Core use

```python
import PythonAutomataLibrary as pal
```

PAL starts in **safe mode**. Safe mode checks public operations and provides useful errors. To select fast mode:

```python
pal.FastMode()
```

Call `FastMode()` before constructing any PAL model or window object. The first such construction locks the process-wide mode. Fast mode removes safety checks from performance-sensitive paths; it is intended for models already validated in safe mode.

Use `@pal.njit` for model computation:

```python
@pal.njit
def Step(grid: pal.AgentGrid):
    ...
```

`pal.njit` wraps Numba `njit` and additionally transforms PAL model code. Among other PAL-specific compilation behavior, annotated PAL arguments allow safe-mode errors in compiled model code to report the model source line.

## Construction

| Function | Purpose |
| --- | --- |
| `NewGrid(dimensions, dtype)` | Typed lattice values. |
| `NewAgentGrid(dimensions, numAgentProps=0, isStackable=False)` | Discrete or continuous agents with optional per-agent properties. |
| `NewPopGrid(dimensions, capacity=None)` | Integer population counts on a lattice. |
| `NewPDEgrid(dimensions)` | Continuous scalar field with diffusion/advection operations. |
| `NewIList()` | Reusable integer query/list container. |
| `NewMultinomial(other=None)` | Binomial/multinomial sampler. |

Spatial grids use one to three nonzero integer dimensions. Each signed dimension must fit the int32 range (excluding its minimum value), and the product of absolute axis lengths must not exceed the int32 maximum. Fractional, nonfinite, zero, oversized, or malformed dimension inputs raise `ValueError` during construction. A **negative dimension enables wrapping on that axis** while its absolute value gives the size. `AgentGrid` alone also accepts `()` for a zero-dimensional nonspatial population:

```python
grid = pal.NewAgentGrid((100, 100))    # 100 x 100, no wrapping
grid = pal.NewAgentGrid((-100, 100))   # x wraps, y does not
grid = pal.NewPDEgrid((-100, -100))    # x and y wrap
```

`numAgentProps` is the number of per-agent floating-point properties. It must be a nonnegative integer fitting AgentGrid's int32 metadata budget (which also reserves entries for spatial dimensions); fractional, nonfinite, negative, or oversized values raise `ValueError`. `isStackable` accepts booleans or `0`/`1`; other values raise `ValueError`. When false, at most one agent occupies each lattice site. `PopGrid.capacity` is an optional nonnegative total-population limit.

`NewMultinomial(other)` accepts another PAL Multinomial in the **same safety mode** and copies its solver configuration, not its current sampling state. An unrelated object raises `TypeError`; mixing safe and fast samplers raises `ValueError`. Create a new sampler without `other` to use the default solver.

## Shared spatial API

`Grid`, `AgentGrid`, `PopGrid`, and `PDEgrid` deliberately share spatial conventions. Geometry metadata (`xDim`, `yDim`, `zDim`, `nDims`, `wrapX`, `wrapY`, `wrapZ`) is read as attributes; geometry operations such as `ToI(...)` and `Box(...)` are methods.

### Geometry

| Member | Meaning |
| --- | --- |
| `len(grid)` | Number of lattice sites. |
| `xDim`, `yDim`, `zDim` | Axis sizes. Missing axes are invalid in safe mode. |
| `nDims` | Number of spatial dimensions. |
| `wrapX`, `wrapY`, `wrapZ` | Whether each axis wraps. |
| `ToI(x, y=-1, z=-1)` | Convert coordinates to a linear lattice index. |
| `ItoX(i)`, `ItoY(i)`, `ItoZ(i)` | Recover coordinates from a linear index. |

The same method names are used in 1D, 2D, and 3D. Supply only the coordinates that exist for the grid.

### Regions and neighborhoods

```python
grid.Box(x1, x2)
grid.Box(x1, x2, y1, y2)
grid.Box(x1, x2, y1, y2, z1, z2)

grid.Hood(hood, x)
grid.Hood(hood, x, y)
grid.Hood(hood, x, y, z)
```

`Box` iterates coordinates in the half-open rectangular region `[x1, x2)`, `[y1, y2)`, `[z1, z2)` for the dimensions supplied. It yields an integer in 1D, `(x, y)` in 2D, and `(x, y, z)` in 3D. Bounds must be int32 integers, and the number of bound pairs must match grid dimensionality. On wrapped axes, out-of-range coordinates are wrapped; on nonwrapped axes, they are skipped. A box wider than a wrapped axis can therefore yield repeated sites. `Hood` maps a tuple of relative-offset tuples around a lattice position to coordinates (an integer in 1D, `(x, y)` in 2D, `(x, y, z)` in 3D), preserving offset order. Wrapped coordinates are mapped into the grid, nonwrapped out-of-range coordinates are omitted, and duplicate coordinates are retained. Offset tuple lengths must match the supplied coordinate count; invalid neighborhood shapes raise an error.

Built-in neighborhoods:

```python
pal.MooreHood(dim, excludeCenter=False)
pal.VonNeumannHood(dim, excludeCenter=False)
pal.CircleHood(dim, rad, excludeCenter=False)
```

`MooreHood` contains offsets with each coordinate in `[-1, 1]`. `VonNeumannHood` contains the center and axis-adjacent offsets. `CircleHood` contains integer offsets whose Euclidean distance from the center is at most `rad`; `rad` may be noninteger, but must be finite and nonnegative. Set `excludeCenter=True` to omit the zero offset. All three constructors support dimensions 1–3 and reject other dimension counts with `ValueError`. Each returns a tuple of integer offset tuples; these describe relative positions, not site indices, until mapped through `grid.Hood(...)`.

## Grid

```python
grid = pal.NewGrid((xDim, yDim), dtype)
```

Supported dtypes are `bool`, signed/unsigned 8/16/32/64-bit integers, and 32/64-bit floats.

`Grid` supports linear indexing, coordinate indexing, and NumPy-like slices:

```python
grid[i]
grid[x, y]
grid[x, y, z]
grid[:] = 0
region = grid[2:5, 3:8]
```

Slice reads return detached NumPy copies. Scalar and slice reads/writes work in Python and compiled PAL model code. In safe mode, invalid scalar indices and coordinates are rejected; fast mode assumes valid access. Safe mode validates bounds, dimensionality, and whether assigned values are representable by the Grid dtype: boolean grids accept only `0`/`1`; integer grids reject fractional, nonfinite, and out-of-range assignments; floating grids reject nonfinite or out-of-range assignments. Use a wider dtype when a calculation may exceed its current range. A detached slice copy is not a live view of the Grid.

## AgentGrid

```python
agents = pal.NewAgentGrid((xDim, yDim), numAgentProps=2)
```

PAL represents an agent by an integer handle. Agent properties are stored on the grid and accessed with `agents[agent, property]`.

### Population and lifecycle

| Operation | Meaning |
| --- | --- |
| `GetPop()` | Number of living agents. |
| `Alive(agent)` | Whether an agent handle is alive. |
| `NewAgentSQ(...)` | Create an agent at a lattice site. |
| `NewAgent(...)` | Create an agent at a continuous position. |
| `Dispose(agent)` | Remove an agent. |
| `All(shuffle=False)` | Snapshot of all living agents; optionally shuffled. |

`All()` has **snapshot semantics**. The returned collection can be iterated while agents are created or disposed without changing the current iteration. `All(shuffle=True)` returns the same population in randomized order.

```python
for agent in agents.All():
    if ShouldDie(agent):
        agents.Dispose(agent)
```

### Position

Discrete/lattice position:

```python
agents.I(agent)
agents.XSQ(agent)
agents.YSQ(agent)
agents.ZSQ(agent)
agents.MoveSQ(agent, x, y, z)
```

Continuous position:

```python
agents.X(agent)
agents.Y(agent)
agents.Z(agent)
agents.Move(agent, x, y, z)
```

`NewAgentSQ` and `MoveSQ` accept either a linear lattice index or lattice coordinates in every spatial dimension. SQ placement uses the center of the selected cell: after `NewAgentSQ(x, y)`, for example, `XSQ(agent)==x` and `YSQ(agent)==y`, while `X(agent)==x+0.5` and `Y(agent)==y+0.5`. Continuous `NewAgent` and `Move` instead keep the supplied spatial coordinates, with `XSQ`/`YSQ`/`ZSQ` identifying the containing cell.

### Occupancy and queries

```python
agents.LastAgent(x, y, z)
agents.AgentsAt(x, y, z)
agents.counts[x, y, z]
```

`LastAgent` returns the most recently stacked agent at a site, or `-1` if the site is empty. Iterate with `for agent in agents.AgentsAt(x, y):` in Python or inside `@pal.njit`. Do not structurally modify the AgentGrid during iteration. Safe mode detects structural changes and raises `RuntimeError`; fast mode omits that check. In Python, `AgentsAt(i)` also accepts a linear site index, including for multidimensional grids; `AgentsAt(x, y)` and `AgentsAt(x, y, z)` use coordinates. This overload is distinct from the continuous-position `AgentsInRadius` query. `counts` exposes lattice occupancy counts and supports linear-site, coordinate, and slice indexing (`agents.counts[i]`, `agents.counts[x, y]`, or `agents.counts[:, :]`). It is a **read-only query view**, not a second mutable population grid. Slice queries return detached arrays of occupancy counts. A nonspatial (`dimensions=()`) AgentGrid has no lattice counts; querying `len(agents.counts)` raises `ValueError`.

By default an `AgentGrid` is not stackable: at most one agent may occupy a lattice site. Set `isStackable=True` when multiple agents per site are required. Safe mode rejects creation or movement into an occupied site on a nonstackable grid without changing model state.

### Radius iteration

`AgentsInRadius(rad, x[, y[, z]], exclude=None)` iterates agents near a continuous point in Python or inside `@pal.njit`. The yielded tuple depends on dimensionality:

```python
for agent, dx in agents1.AgentsInRadius(rad, x):
    ...
for agent, dx, dy, distSq in agents2.AgentsInRadius(rad, x, y):
    ...
for agent, dx, dy, dz, distSq in agents3.AgentsInRadius(rad, x, y, z):
    ...
```

Displacements account for wrapped axes; `distSq` is squared Euclidean distance in 2D/3D. `exclude` omits a specified agent handle. As with Python-side `AgentsAt`, structural modification during iteration is unsupported, and the generation check is enabled only in safe mode.

### Wrapping

Discrete coordinates:

```python
InWrapSQX(value)
InWrapSQY(value)
InWrapSQZ(value)
```

Continuous coordinates:

```python
InWrapX(value)
InWrapY(value)
InWrapZ(value)
```

Wrapped displacement:

```python
DispWrapX(x1, x2)
DispWrapY(y1, y2)
DispWrapZ(z1, z2)
```

The displacement methods return the shortest signed displacement accounting for periodic boundaries.

## PopGrid

```python
pop = pal.NewPopGrid((xDim, yDim), capacity=None)
```

`PopGrid` stores integer population counts. Indexing and direct assignment modify the current population immediately.

```python
pop[x, y]
pop[x, y] = 10
pop[:] = 0
```

Linear (`pop[:]`) and spatial (`pop[:, :]`) slice reads return detached `int64` NumPy arrays; slice assignment writes to the population. Population changes can instead be accumulated and applied together:

```python
pop.Add(delta, x, y)
pop.Update()
```

`Add` changes the pending delta, not the current value: `GetPop()`, indexing, and `All()` continue to report the current population until `Update()`. `Update()` applies all pending changes simultaneously. This is useful when a timestep should not depend on iteration order.

| Operation | Meaning |
| --- | --- |
| `GetPop()` | Total population over all sites. |
| `All()` | Copy of the linear indices of currently nonzero sites. |
| `Reset()` | Clear current populations and pending changes. |
| `InWrapX/Y/Z(value)` | Wrap a coordinate on the corresponding axis. |

`capacity=None` uses the int64 maximum as the population cap. An explicit `capacity` must be a nonnegative int64 integer; invalid, fractional, nonfinite, or out-of-range values raise `ValueError` during construction. Capacity constrains the **total population**, not an independent limit per site. In safe mode, test births and transfers against capacity and nonnegative-count constraints before switching to fast mode.

## PDEgrid

```python
field = pal.NewPDEgrid((xDim, yDim))
```

`PDEgrid` stores a continuous `float32` scalar field. Direct indexing/assignment changes the current field immediately. Slice reads return detached `float32` NumPy arrays; array slice assignment writes to the field, but editing a previously read slice does not. Linear (`field[:]`) and spatial (`field[:, :]`) slices are supported. `Add(value, ...)` accumulates a pending delta; `Update()` applies pending changes simultaneously. `Reset()` clears both the field and pending changes.

### Space and time

```python
field.SetTimeSpaceStep(dt, dx, dy=1.0, dz=1.0)
field.Dt()
field.Dx()
field.Dy()
field.Dz()
```

Transport methods use these spacings. `Dy()` and `Dz()` apply only when those dimensions exist; safe mode rejects missing-dimension access. `dt`, `dx`, `dy`, and `dz` must be finite and positive in safe mode.

### Diffusion

```python
field.Diffusion(rateConstant, ...)
field.DiffusionMask(rateConstant, mask)
field.DiffusionField(rateConstants, ...)
field.DiffusionInterfaces(rateConstantsX, rateConstantsY=None, rateConstantsZ=None, ...)
field.DiffusionADI(rateConstant, ...)
field.DiffusionRadialCircle(rateConstant, outerBC=None)
field.DiffusionRadialSphere(rateConstant, outerBC=None)
```

`Diffusion` uses one constant diffusion rate. `DiffusionField` uses spatially varying rates. `DiffusionInterfaces` supplies rates on cell interfaces. `DiffusionMask` restricts diffusion with a mask. `DiffusionADI` provides the alternating-direction implicit solver. Radial methods solve the corresponding radially symmetric circle/sphere geometry.

Cartesian diffusion methods accept optional `xMinBC`, `xMaxBC`, `yMinBC`, `yMaxBC`, `zMinBC`, and `zMaxBC` boundary values. Wrapped axes use periodic boundaries. Field/interface arrays must match the grid geometry and dimensionality; safe mode validates these preconditions before changing pending state.

### Advection

```python
field.Advection(vx, vy=0.0, vz=0.0, ...)
field.AdvectionField(xVels, yVels=None, zVels=None, ...)
field.AdvectionInterfaces(xVels, yVels=None, zVels=None, ...)
```

`Advection` uses constant velocity components. `AdvectionField` uses spatial velocity fields. `AdvectionInterfaces` supplies velocities at cell interfaces. Supply velocity components only for dimensions that exist. The same optional Cartesian boundary arguments used by diffusion are available.

## IList

```python
items = pal.NewIList()
```

`IList` is PAL's reusable integer query/list container. Stored values are nonnegative int32 integers in the supported contract.

| Operation | Meaning |
| --- | --- |
| `Append(i)` | Append a nonnegative int32 value; returns the IList for chaining. |
| `Clear()` | Remove all entries; returns the IList for chaining. |
| `Random()` | Return a random entry. |
| `Shuffle()` | Shuffle entries in place; returns the IList for chaining. |
| `All()` | Return a detached copy. |
| `Iter()` | Return a no-copy/live iterable view. |
| `len(items)`, `items[i]` | Length/index access. |

Use `Iter()` for one-pass iteration when a copy is unnecessary. Use `All()` when the returned values must remain independent of subsequent IList changes.

## Random numbers and Multinomial

```python
pal.Seed(seed)
pal.Random()       # uniform float
pal.RandInt(max)  # integer in [0, max)
```

`Seed` accepts integral seeds in the uint64 domain (`0` through `2**64 - 1`); invalid or nonfinite inputs raise `ValueError`. `RandInt(max)` requires a positive integer in the int64 domain and returns an integer in `[0, max)`; zero, negative, fractional, nonfinite, or out-of-range bounds raise `ValueError`. `Random()` returns a uniform floating-point draw. PAL random functions share one seeded stream across Python and compiled PAL calls, so reseeding reproduces the same call sequence across that boundary. The stream is call-order dependent: adding or removing a random draw changes later results. This does not seed NumPy's separate RNG.

For binomial/multinomial sampling:

```python
multi = pal.NewMultinomial()
x = multi.Binomial(n, p)

multi.Setup(n)
a = multi.Sample(pA)
b = multi.Sample(pB)
# remaining count/probability mass stays in the sampler
```

`Setup(n)` initializes the remaining count for a multinomial sequence and is chainable. Each `Sample(p)` consumes count and probability mass; in safe mode cumulative requested probability cannot exceed the remaining mass. `Binomial(n, p)` performs an independent binomial draw. `NewMultinomial(other)` reuses the other sampler's underlying solver configuration but starts with fresh multinomial sampling state.

## Visualization

PAL provides a pixel renderer for lattice models and an OpenGL renderer for 2D/3D geometry. Rendering runs separately from model computation.

### PixWindow

```python
pix, window = pal.StartPixWindow(xDim, yDim, scale=1, title="PAL", headless=False)
```

`pix` is a write-oriented pixel buffer, not a general-purpose numeric `Grid`: it supports assignment but does not expose a public pixel-read API. Assign integer RGB colors by coordinate or linear index:

```python
pix[x, y] = 0xFF0000
pix[i] = 0x00FF00
```

It also provides `Xdim()`, `Ydim()`, `ToI(x, y)`, `ItoX(i)`, `ItoY(i)`, and `len(pix)`.

The window lifecycle is:

```python
window.Update()
window.Save(path, block=False)
window.StartGif(path, delay=100)
window.AddGifFrame(block=False)
window.StopGif()
window.IsOpen()
window.Close()
```

`Update()` publishes the current pixel buffer to the renderer. `block=True` on output operations waits for the requested output to complete before returning.

### OpenGLWindow

```python
draw, window = pal.StartOpenGLWindow(xDim, yDim, zDim=None,
                                     width=800, height=800,
                                     title="PAL", headless=False)
```

Draw geometry with:

```python
draw.Circle(rad, color, x, y, z=0.0)
draw.Box(xLen, color, x, y, z=0.0, yLen=None, zLen=None)
draw.BoxSQ(color, x, y, z=0.0)
draw.Line(width, color, x1, y1, x2, y2, z1=0.0, z2=0.0)
```

Window controls include `Borders(width, color)`, `Camera(x, y, z, yaw=None, pitch=None)`, `Background(color)`, `Clear()`, `Update()`, `IsOpen()`, `Save(path, block=False)`, `StartGif(path, delay=100)`, `AddGifFrame(block=False, timeout=30)`, `StopGif(timeout=30)`, and `Close()`.

Colors are packed integer RGB values such as `0xFF0000`. `pal.ColorScale(colors, value)` interpolates RGB channels across a sequence of colors for normalized `value`; values outside `[0, 1]` clamp to the endpoint colors.

Set `headless=True` for off-screen rendering. This is useful for automated image/GIF generation and testing; OpenGL headless rendering requires a supported standalone backend such as EGL.

Use `pal.AwaitWindows()` when model execution should wait for PAL windows to finish their lifecycle.

## Safe and fast contracts

Safe and fast objects expose the same modeling API. PAL starts in safe mode. To opt into fast mode, call `pal.FastMode()` **before constructing any PAL model or window object**. The first such construction locks the process-wide mode selection; a later `pal.FastMode()` raises `RuntimeError`. Run safe and fast comparisons in separate processes rather than trying to switch modes mid-run. Safe mode validates public operations and is the development/default mode; fast mode assumes the model obeys those contracts and removes performance-sensitive checks. Examples of safe-mode checks include lattice bounds and dimensionality, dead-agent and occupancy errors, PopGrid overflow, and PDE field/interface array shape or dimensionality. Invalid operations in fast mode are outside the supported contract rather than an alternative error-handling API.

Use safe mode until the model is correct, then benchmark fast mode. Do not rely on safe-mode exceptions as model control flow.

## Saving PAL objects

PAL native-backed model objects support standard Python pickling. The serialized form stores model state, not process-local native pointers. This allows model state to be saved and restored with Python's `pickle` module.

## Choosing the right structure

| Need | PAL structure |
| --- | --- |
| One typed value per lattice site | `Grid` |
| Individual agents with identity/properties | `AgentGrid` |
| Integer counts without individual identity | `PopGrid` |
| Continuous diffusing/advecting field | `PDEgrid` |
| Reusable integer query result | `IList` |
| Binomial/multinomial draws | `Multinomial` |

These structures are designed to be composed inside the same compiled model step. A model can, for example, query agents, accumulate population counts, update a PDE field, and feed the resulting field back into agent behavior without leaving `@pal.njit`.
