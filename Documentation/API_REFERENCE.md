# PAL API Reference

Generated from public `Protocol` declarations and top-level factory signatures in `NativeCore.py`. Regenerate with `python Documentation/generators/generate_api_reference.py`; use `--check` to detect drift. These declarations describe the annotation/autocomplete surface; consult [API Guide](API_GUIDE.pdf) for behavior and the implementation/tests for runtime semantics.

## Factories and utilities

```python
FastMode()
# Select PAL fast mode before constructing any PAL object or window; the process-wide mode cannot be changed after construction.
NewIList()
# Create an empty PAL integer list usable from Python and supported compiled functions.
MooreHood(dim, excludeCenter=False)
# Build relative integer offsets for a Moore neighborhood in `dim` dimensions; optionally exclude the center offset.
VonNeumannHood(dim, excludeCenter=False)
# Build relative integer offsets for a Von Neumann neighborhood in `dim` dimensions; optionally exclude the center offset.
CircleHood(dim, rad, excludeCenter=False)
# Build relative integer offsets within radius `rad` in `dim` dimensions; optionally exclude the center offset.
NewMultinomial(other=None)
# Create a PAL multinomial/binomial sampler, optionally reusing configuration from `other`.
NewGrid(dimensions, dtype)
# Create a typed lattice grid with the requested dimensions and element dtype.
NewAgentGrid(dimensions, numAgentProps=0, isStackable=False)
# Create a grid for individual agents; `numAgentProps` sets the number of per-agent properties and `isStackable` controls whether multiple agents may occupy a site.
NewPopGrid(dimensions, capacity=None)
# Create a lattice of integer population counts, optionally limiting the count at each site with `capacity`.
NewPDEgrid(dimensions)
# Create a lattice-backed scalar field for numerical transport operations.
```

## Multinomial

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `Binomial(self, n: int, p: float) -> int` | Description not yet provided. |
| method | `Setup(self, n: int)` | Configure the sampler for `n` trials before repeated sampling. |
| method | `Sample(self, p: float) -> int` | Description not yet provided. |

## IList

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__len__(self) -> int` | Description not yet provided. |
| method | `__getitem__(self, index: int) -> int` | Description not yet provided. |
| method | `Append(self, i: int)` | Append integer `i` to the end of the list. |
| method | `Clear(self)` | Remove all values from the list. |
| method | `Random(self) -> int` | Description not yet provided. |
| method | `Shuffle(self)` | Shuffle the list in place using PAL's random stream. |
| method | `All(self) -> np.ndarray` | Description not yet provided. |
| method | `Iter(self) -> np.ndarray` | Description not yet provided. |

## AgentGrid

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__len__(self) -> int` | Description not yet provided. |
| property | `xDim(self) -> int` | Description not yet provided. |
| property | `yDim(self) -> int` | Description not yet provided. |
| property | `zDim(self) -> int` | Description not yet provided. |
| property | `nDims(self) -> int` | Description not yet provided. |
| property | `wrapX(self) -> bool` | Description not yet provided. |
| property | `wrapY(self) -> bool` | Description not yet provided. |
| property | `wrapZ(self) -> bool` | Description not yet provided. |
| method | `Box(self, x1: int, x2: int, y1: int=None, y2: int=None, z1: int=None, z2: int=None)` | Iterate lattice coordinates in the half-open rectangular region [x1, x2) × [y1, y2) × [z1, z2); omit unused axes in lower dimensions. |
| method | `Hood(self, hood, x: int, y: int=None, z: int=None, *, unroll: bool=False)` | Apply relative neighborhood offsets at a lattice coordinate, respecting wrapped and nonwrapped boundaries. `unroll=True` enables compiled-loop expansion where supported. |
| method | `GetPop(self) -> int` | Description not yet provided. |
| method | `Alive(self, agent: int) -> bool` | Description not yet provided. |
| method | `NewAgentSQ(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `NewAgent(self, x: float, y: float=-1.0, z: float=-1.0) -> int` | Description not yet provided. |
| method | `Dispose(self, agent: int)` | Remove a living agent from the grid; its handle must not be used as a live agent afterward. |
| method | `I(self, agent: int) -> int` | Description not yet provided. |
| method | `XSQ(self, agent: int) -> int` | Description not yet provided. |
| method | `YSQ(self, agent: int) -> int` | Description not yet provided. |
| method | `ZSQ(self, agent: int) -> int` | Description not yet provided. |
| method | `X(self, agent: int) -> float` | Description not yet provided. |
| method | `Y(self, agent: int) -> float` | Description not yet provided. |
| method | `Z(self, agent: int) -> float` | Description not yet provided. |
| method | `__getitem__(self, key) -> float` | Description not yet provided. |
| method | `__setitem__(self, key, value: float)` | Set a supported per-agent property value for the selected agent/property key. |
| method | `ToI(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `ItoX(self, i: int) -> int` | Description not yet provided. |
| method | `ItoY(self, i: int) -> int` | Description not yet provided. |
| method | `ItoZ(self, i: int) -> int` | Description not yet provided. |
| method | `InWrapSQX(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapSQY(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapSQZ(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapX(self, value: float) -> float` | Description not yet provided. |
| method | `InWrapY(self, value: float) -> float` | Description not yet provided. |
| method | `InWrapZ(self, value: float) -> float` | Description not yet provided. |
| method | `DispWrapX(self, x1: float, x2: float) -> float` | Description not yet provided. |
| method | `DispWrapY(self, y1: float, y2: float) -> float` | Description not yet provided. |
| method | `DispWrapZ(self, z1: float, z2: float) -> float` | Description not yet provided. |
| property | `counts(self) -> object` | Description not yet provided. |
| method | `LastAgent(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `MoveSQ(self, agent: int, x: int, y: int=-1, z: int=-1)` | Move an agent to a lattice site specified by integer coordinates or a supported linear site index. |
| method | `Move(self, agent: int, x: float, y: float=-1.0, z: float=-1.0)` | Move an agent to continuous coordinates; use `MoveSQ` when movement should target a lattice site. |
| method | `AgentsAt(self, x: int, y: int=-1, z: int=-1)` | Return the agent handle or handles occupying the specified lattice site. |
| method | `AgentsInRadius(self, rad: float, x: float, y: float=-1.0, z: float=-1.0, *, exclude: int=None)` | Return agents within radius `rad` of a continuous position; `exclude` can omit one agent from the results. |
| method | `All(self, shuffle: bool=False) -> np.ndarray` | Description not yet provided. |

## Grid

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__len__(self) -> int` | Description not yet provided. |
| method | `__getitem__(self, key)` | Read the value or state selected by the supported index or coordinate key. |
| method | `__setitem__(self, key, value)` | Assign the value or state selected by the supported index or coordinate key. |
| property | `xDim(self) -> int` | Description not yet provided. |
| property | `yDim(self) -> int` | Description not yet provided. |
| property | `zDim(self) -> int` | Description not yet provided. |
| property | `nDims(self) -> int` | Description not yet provided. |
| property | `wrapX(self) -> bool` | Description not yet provided. |
| property | `wrapY(self) -> bool` | Description not yet provided. |
| property | `wrapZ(self) -> bool` | Description not yet provided. |
| method | `Box(self, x1: int, x2: int, y1: int=None, y2: int=None, z1: int=None, z2: int=None)` | Iterate lattice coordinates in the half-open rectangular region [x1, x2) × [y1, y2) × [z1, z2); omit unused axes in lower dimensions. |
| method | `Hood(self, hood, x: int, y: int=None, z: int=None, *, unroll: bool=False)` | Apply relative neighborhood offsets at a lattice coordinate, respecting wrapped and nonwrapped boundaries. `unroll=True` enables compiled-loop expansion where supported. |
| method | `ToI(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `ItoX(self, i: int) -> int` | Description not yet provided. |
| method | `ItoY(self, i: int) -> int` | Description not yet provided. |
| method | `ItoZ(self, i: int) -> int` | Description not yet provided. |
| method | `InWrapX(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapY(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapZ(self, value: int) -> int` | Description not yet provided. |

## PopGrid

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__len__(self) -> int` | Description not yet provided. |
| property | `xDim(self) -> int` | Description not yet provided. |
| property | `yDim(self) -> int` | Description not yet provided. |
| property | `zDim(self) -> int` | Description not yet provided. |
| property | `nDims(self) -> int` | Description not yet provided. |
| property | `wrapX(self) -> bool` | Description not yet provided. |
| property | `wrapY(self) -> bool` | Description not yet provided. |
| property | `wrapZ(self) -> bool` | Description not yet provided. |
| method | `Box(self, x1: int, x2: int, y1: int=None, y2: int=None, z1: int=None, z2: int=None)` | Iterate lattice coordinates in the half-open rectangular region [x1, x2) × [y1, y2) × [z1, z2); omit unused axes in lower dimensions. |
| method | `Hood(self, hood, x: int, y: int=None, z: int=None, *, unroll: bool=False)` | Apply relative neighborhood offsets at a lattice coordinate, respecting wrapped and nonwrapped boundaries. `unroll=True` enables compiled-loop expansion where supported. |
| method | `ToI(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `ItoX(self, i: int) -> int` | Description not yet provided. |
| method | `ItoY(self, i: int) -> int` | Description not yet provided. |
| method | `ItoZ(self, i: int) -> int` | Description not yet provided. |
| method | `InWrapX(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapY(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapZ(self, value: int) -> int` | Description not yet provided. |
| method | `__getitem__(self, key) -> int` | Description not yet provided. |
| method | `__setitem__(self, key, value: int)` | Set the population count at the selected site immediately; this is distinct from queueing a delta with `Add()`. |
| method | `Add(self, value: int, x: int, y: int=-1, z: int=-1)` | Queue a signed population-count delta at a site; call `Update()` to apply pending deltas. |
| method | `Update(self)` | Apply queued population-count deltas to the current counts. |
| method | `Reset(self)` | Clear population counts and any pending deltas. |
| method | `GetPop(self) -> int` | Description not yet provided. |
| method | `All(self) -> np.ndarray` | Description not yet provided. |

## PDEgrid

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__len__(self) -> int` | Description not yet provided. |
| property | `xDim(self) -> int` | Description not yet provided. |
| property | `yDim(self) -> int` | Description not yet provided. |
| property | `zDim(self) -> int` | Description not yet provided. |
| property | `nDims(self) -> int` | Description not yet provided. |
| property | `wrapX(self) -> bool` | Description not yet provided. |
| property | `wrapY(self) -> bool` | Description not yet provided. |
| property | `wrapZ(self) -> bool` | Description not yet provided. |
| method | `Box(self, x1: int, x2: int, y1: int=None, y2: int=None, z1: int=None, z2: int=None)` | Iterate lattice coordinates in the half-open rectangular region [x1, x2) × [y1, y2) × [z1, z2); omit unused axes in lower dimensions. |
| method | `Hood(self, hood, x: int, y: int=None, z: int=None, *, unroll: bool=False)` | Apply relative neighborhood offsets at a lattice coordinate, respecting wrapped and nonwrapped boundaries. `unroll=True` enables compiled-loop expansion where supported. |
| method | `ToI(self, x: int, y: int=-1, z: int=-1) -> int` | Description not yet provided. |
| method | `ItoX(self, i: int) -> int` | Description not yet provided. |
| method | `ItoY(self, i: int) -> int` | Description not yet provided. |
| method | `ItoZ(self, i: int) -> int` | Description not yet provided. |
| method | `InWrapX(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapY(self, value: int) -> int` | Description not yet provided. |
| method | `InWrapZ(self, value: int) -> int` | Description not yet provided. |
| method | `__getitem__(self, key) -> float` | Description not yet provided. |
| method | `__setitem__(self, key, value: float)` | Set the field value at the selected site immediately; this is distinct from queueing a change with `Add()`. |
| method | `Add(self, value: float, x: int, y: int=-1, z: int=-1)` | Queue a field-value delta at a site; call `Update()` to apply pending changes. |
| method | `Update(self)` | Apply queued field changes to the current field. |
| method | `Reset(self)` | Clear the field and any pending changes. |
| method | `SetTimeSpaceStep(self, dt: float, dx: float, dy: float=1.0, dz: float=1.0)` | Set the temporal step `dt` and spatial steps `dx`, `dy`, and `dz` used by numerical transport methods. |
| method | `Dx(self) -> float` | Description not yet provided. |
| method | `Dy(self) -> float` | Description not yet provided. |
| method | `Dz(self) -> float` | Description not yet provided. |
| method | `Dt(self) -> float` | Description not yet provided. |
| method | `Diffusion(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply scalar-coefficient diffusion using the configured grid spacing and optional per-face boundary conditions. |
| method | `DiffusionMask(self, rateConstant: float, mask=None)` | Apply diffusion only across interfaces allowed by `mask`. |
| method | `DiffusionField(self, rateConstants, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply diffusion with spatially varying rate constants and optional boundary conditions. |
| method | `DiffusionInterfaces(self, rateConstantsX, rateConstantsY=None, rateConstantsZ=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply diffusion using rate constants defined on grid interfaces; separate arrays may specify x, y, and z interfaces. |
| method | `DiffusionADI(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply diffusion using an alternating-direction implicit method. |
| method | `DiffusionRadialCircle(self, rateConstant: float, outerBC=None)` | Apply radial diffusion on a circular domain with an optional outer boundary condition. |
| method | `DiffusionRadialSphere(self, rateConstant: float, outerBC=None)` | Apply radial diffusion on a spherical domain with an optional outer boundary condition. |
| method | `Advection(self, vx: float, vy: float=0.0, vz: float=0.0, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply advection using a constant velocity vector and optional per-face boundary conditions. |
| method | `AdvectionField(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply advection using spatially varying velocity fields. |
| method | `AdvectionInterfaces(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` | Apply advection using velocity values defined on grid interfaces. |

## PixWindow visualization

```python
StartPixWindow(xDim, yDim, scale=1, title='PAL', headless=False) -> tuple[Pix, PixWindow]
```

Description not yet provided.

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `IsOpen(self) -> bool` | Description not yet provided. |
| method | `Update(self)` | Description not yet provided. |
| method | `Save(self, path: str, block=False)` | Description not yet provided. |
| method | `StartGif(self, path: str, delay=100)` | Description not yet provided. |
| method | `AddGifFrame(self, block=False)` | Description not yet provided. |
| method | `StopGif(self)` | Description not yet provided. |
| method | `Close(self)` | Description not yet provided. |

## OpenGLWindow visualization

```python
StartOpenGLWindow(xDim, yDim, zDim=None, width=800, height=800, title='PAL', headless=False) -> tuple[OpenGLDraw, OpenGLWindow]
```

Description not yet provided.

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `Update(self)` | Description not yet provided. |
| method | `IsOpen(self) -> bool` | Description not yet provided. |
| method | `StartGif(self, path, delay=100)` | Description not yet provided. |
| method | `AddGifFrame(self, block=False, timeout=30)` | Description not yet provided. |
| method | `StopGif(self, timeout=30)` | Description not yet provided. |
| method | `Circle(self, rad, color, x, y, z=None)` | Description not yet provided. |
| method | `Box(self, xLen, color, x, y, z=None, yLen=None, zLen=None)` | Description not yet provided. |
| method | `BoxSQ(self, color, x, y, z=None)` | Description not yet provided. |
| method | `Line(self, width, color, x1, y1, x2, y2, z1=None, z2=None)` | Description not yet provided. |
| method | `Borders(self, width, color)` | Description not yet provided. |
| method | `Camera(self, x, y, z, yaw=None, pitch=None)` | Description not yet provided. |
| method | `Clear(self)` | Description not yet provided. |
| method | `Background(self, color)` | Description not yet provided. |
| method | `Save(self, path: str, block=False)` | Description not yet provided. |
| method | `Close(self)` | Description not yet provided. |

The `OpenGLWindow` table follows its source Protocol, which also contains method implementations; repeated declarations are listed once. `OpenGLDraw` is a compiled primitive collector, while the returned window handles rendering and output. Consult the [API Guide](API_GUIDE.pdf) for which object to use in Python and compiled drawing code.

## Pix drawing buffer

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `__setitem__(self, key, color)` | Description not yet provided. |
| method | `Xdim(self)` | Description not yet provided. |
| method | `Ydim(self)` | Description not yet provided. |
| method | `ToI(self, x, y)` | Description not yet provided. |
| method | `ItoX(self, i)` | Description not yet provided. |
| method | `ItoY(self, i)` | Description not yet provided. |
| method | `__len__(self)` | Description not yet provided. |

## OpenGLDraw primitive collector

| Kind | Declaration | Description |
| --- | --- | --- |
| method | `Clear(self)` | Description not yet provided. |
| method | `Circle(self, rad, color, x, y, z=0.0)` | Description not yet provided. |
| method | `Box(self, xLen, color, x, y, z=0.0, yLen=-1.0, zLen=-1.0)` | Description not yet provided. |
| method | `BoxSQ(self, color, x, y, z=0.0)` | Description not yet provided. |
| method | `Line(self, width, color, x1, y1, x2, y2, z1=0.0, z2=0.0)` | Description not yet provided. |

## Scope and validation

This reference lists public Protocol signatures, not internal safe/fast jitclass methods. A signature does not capture every runtime overload, indexing form, or validation rule. See the API Guide and executable tests. Visualization constructors and window Protocol methods are included; Pix buffer and OpenGLDraw method signatures are also included; consult the API Guide for behavior.
