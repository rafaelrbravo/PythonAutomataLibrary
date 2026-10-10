# PAL API Reference

Generated from public `Protocol` declarations and top-level factory signatures in `NativeCore.py`. Regenerate with `python Documentation/generate_api_reference.py`; use `--check` to detect drift. These declarations describe the annotation/autocomplete surface; consult [API Guide](API_GUIDE.md) for behavior and the implementation/tests for runtime semantics.

## Factories and utilities

```python
FastMode()
NewIList()
MooreHood(dim, excludeCenter=False)
VonNeumannHood(dim, excludeCenter=False)
CircleHood(dim, rad, excludeCenter=False)
NewMultinomial(other=None)
NewGrid(dimensions, dtype)
NewAgentGrid(dimensions, numAgentProps=0, isStackable=False)
NewPopGrid(dimensions, capacity=None)
NewPDEgrid(dimensions)
```

## Multinomial

| Kind | Declaration |
| --- | --- |
| method | `Binomial(self, n: int, p: float) -> int` |
| method | `Setup(self, n: int)` |
| method | `Sample(self, p: float) -> int` |

## IList

| Kind | Declaration |
| --- | --- |
| method | `__len__(self) -> int` |
| method | `__getitem__(self, index: int) -> int` |
| method | `Append(self, i: int)` |
| method | `Clear(self)` |
| method | `Random(self) -> int` |
| method | `Shuffle(self)` |
| method | `All(self) -> np.ndarray` |
| method | `Iter(self) -> np.ndarray` |

## AgentGrid

| Kind | Declaration |
| --- | --- |
| method | `__len__(self) -> int` |
| property | `xDim(self) -> int` |
| property | `yDim(self) -> int` |
| property | `zDim(self) -> int` |
| property | `nDims(self) -> int` |
| property | `wrapX(self) -> bool` |
| property | `wrapY(self) -> bool` |
| property | `wrapZ(self) -> bool` |
| method | `Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None)` |
| method | `Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False)` |
| method | `GetPop(self) -> int` |
| method | `Alive(self, agent: int) -> bool` |
| method | `NewAgentSQ(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `NewAgent(self, x: float, y: float = -1.0, z: float = -1.0) -> int` |
| method | `Dispose(self, agent: int)` |
| method | `I(self, agent: int) -> int` |
| method | `XSQ(self, agent: int) -> int` |
| method | `YSQ(self, agent: int) -> int` |
| method | `ZSQ(self, agent: int) -> int` |
| method | `X(self, agent: int) -> float` |
| method | `Y(self, agent: int) -> float` |
| method | `Z(self, agent: int) -> float` |
| method | `__getitem__(self, key) -> float` |
| method | `__setitem__(self, key, value: float)` |
| method | `ToI(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `ItoX(self, i: int) -> int` |
| method | `ItoY(self, i: int) -> int` |
| method | `ItoZ(self, i: int) -> int` |
| method | `InWrapSQX(self, value: int) -> int` |
| method | `InWrapSQY(self, value: int) -> int` |
| method | `InWrapSQZ(self, value: int) -> int` |
| method | `InWrapX(self, value: float) -> float` |
| method | `InWrapY(self, value: float) -> float` |
| method | `InWrapZ(self, value: float) -> float` |
| method | `DispWrapX(self, x1: float, x2: float) -> float` |
| method | `DispWrapY(self, y1: float, y2: float) -> float` |
| method | `DispWrapZ(self, z1: float, z2: float) -> float` |
| property | `counts(self) -> object` |
| method | `LastAgent(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `MoveSQ(self, agent: int, x: int, y: int = -1, z: int = -1)` |
| method | `Move(self, agent: int, x: float, y: float = -1.0, z: float = -1.0)` |
| method | `AgentsAt(self, x: int, y: int = -1, z: int = -1)` |
| method | `All(self, shuffle: bool = False) -> np.ndarray` |

## Grid

| Kind | Declaration |
| --- | --- |
| method | `__len__(self) -> int` |
| method | `__getitem__(self, key)` |
| method | `__setitem__(self, key, value)` |
| property | `xDim(self) -> int` |
| property | `yDim(self) -> int` |
| property | `zDim(self) -> int` |
| property | `nDims(self) -> int` |
| property | `wrapX(self) -> bool` |
| property | `wrapY(self) -> bool` |
| property | `wrapZ(self) -> bool` |
| method | `Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None)` |
| method | `Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False)` |
| method | `ToI(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `ItoX(self, i: int) -> int` |
| method | `ItoY(self, i: int) -> int` |
| method | `ItoZ(self, i: int) -> int` |

## PopGrid

| Kind | Declaration |
| --- | --- |
| method | `__len__(self) -> int` |
| property | `xDim(self) -> int` |
| property | `yDim(self) -> int` |
| property | `zDim(self) -> int` |
| property | `nDims(self) -> int` |
| property | `wrapX(self) -> bool` |
| property | `wrapY(self) -> bool` |
| property | `wrapZ(self) -> bool` |
| method | `Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None)` |
| method | `Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False)` |
| method | `ToI(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `ItoX(self, i: int) -> int` |
| method | `ItoY(self, i: int) -> int` |
| method | `ItoZ(self, i: int) -> int` |
| method | `InWrapX(self, value: int) -> int` |
| method | `InWrapY(self, value: int) -> int` |
| method | `InWrapZ(self, value: int) -> int` |
| method | `__getitem__(self, key) -> int` |
| method | `__setitem__(self, key, value: int)` |
| method | `Add(self, value: int, x: int, y: int = -1, z: int = -1)` |
| method | `Update(self)` |
| method | `Reset(self)` |
| method | `GetPop(self) -> int` |
| method | `All(self) -> np.ndarray` |

## PDEgrid

| Kind | Declaration |
| --- | --- |
| method | `__len__(self) -> int` |
| property | `xDim(self) -> int` |
| property | `yDim(self) -> int` |
| property | `zDim(self) -> int` |
| property | `nDims(self) -> int` |
| property | `wrapX(self) -> bool` |
| property | `wrapY(self) -> bool` |
| property | `wrapZ(self) -> bool` |
| method | `Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None)` |
| method | `Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False)` |
| method | `ToI(self, x: int, y: int = -1, z: int = -1) -> int` |
| method | `ItoX(self, i: int) -> int` |
| method | `ItoY(self, i: int) -> int` |
| method | `ItoZ(self, i: int) -> int` |
| method | `InWrapX(self, value: int) -> int` |
| method | `InWrapY(self, value: int) -> int` |
| method | `InWrapZ(self, value: int) -> int` |
| method | `__getitem__(self, key) -> float` |
| method | `__setitem__(self, key, value: float)` |
| method | `Add(self, value: float, x: int, y: int = -1, z: int = -1)` |
| method | `Update(self)` |
| method | `Reset(self)` |
| method | `SetTimeSpaceStep(self, dt: float, dx: float, dy: float = 1.0, dz: float = 1.0)` |
| method | `Dx(self) -> float` |
| method | `Dy(self) -> float` |
| method | `Dz(self) -> float` |
| method | `Dt(self) -> float` |
| method | `Diffusion(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `DiffusionMask(self, rateConstant: float, mask=None)` |
| method | `DiffusionField(self, rateConstants, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `DiffusionInterfaces(self, rateConstantsX, rateConstantsY=None, rateConstantsZ=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `DiffusionADI(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `DiffusionRadialCircle(self, rateConstant: float, outerBC=None)` |
| method | `DiffusionRadialSphere(self, rateConstant: float, outerBC=None)` |
| method | `Advection(self, vx: float, vy: float = 0.0, vz: float = 0.0, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `AdvectionField(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |
| method | `AdvectionInterfaces(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None)` |

## Scope and validation

This reference lists public Protocol signatures, not internal safe/fast jitclass methods. A signature does not capture every runtime overload, indexing form, or validation rule. See the API Guide and executable tests. Visualization APIs are not yet included in the Protocol surface and must be documented separately.
