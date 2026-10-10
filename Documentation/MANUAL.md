# PAL Manual

This manual starts with a small working model and then explains how PAL's pieces fit together. Use the [API Guide](API_GUIDE.md) for compact behavior summaries and the [API Reference](API_REFERENCE.md) for signatures.

## 1. The model loop

A PAL model normally has three parts: **Setup** creates state, **Step** changes state, and **Draw** presents it. PAL does not impose a model class or a scheduler. You choose the update order and when to render.

```python
import PythonAutomataLibrary as pal

@pal.njit
def Step(pop: pal.PopGrid):
    # Apply a synchronous increment to every site.
    for i in range(len(pop)):
        pop.Add(1, i)
    pop.Update()

def main():
    pop = pal.NewPopGrid((40, 40))
    for _ in range(10):
        Step(pop)
    print(pop.GetPop())

if __name__ == "__main__":
    main()
```

The example contains 1,600 sites and runs 10 increments per site, so the final total population is 16,000. `Add` accumulates changes; `Update` applies them together. That distinction is useful when one site's update must not affect its neighbors during the same timestep.

Begin in PAL's default **safe mode**. Only select `pal.FastMode()` before constructing any PAL object, after the model is validated and when performance measurements justify it.

## 2. Choose the right representation

A `Grid` holds one typed value per site. An `AgentGrid` holds individually identifiable agents, optionally with per-agent properties. A `PopGrid` holds integer counts without individual identity. A `PDEgrid` holds a continuous scalar field and transport operations. These can be combined: for example, agents consume a PDE field while population counts accumulate in a separate PopGrid.

Use a tuple of dimensions when constructing a spatial grid. Negative dimensions mean periodic wrapping on that axis, so `(-40, 40)` wraps x but not y. `AgentGrid` also supports an empty dimension tuple for a nonspatial population.

## 3. Coordinates and neighborhoods

A grid accepts linear lattice indices. In a 2D grid, `ToI(x, y)` converts coordinates to an index, and `ItoX(i)` and `ItoY(i)` convert back. Grid geometry is exposed as attributes such as `xDim`, `yDim`, `nDims`, and `wrapX`, not methods.

PAL provides `MooreHood`, `VonNeumannHood`, and `CircleHood`. They return relative integer offsets. `grid.Hood(hood, x, y)` maps a neighborhood around a site, taking the grid's periodic boundaries into account. `grid.Box(...)` iterates a half-open rectangular region.

## 4. Individual agents

An agent is an integer handle owned by an `AgentGrid`. Create it with `NewAgentSQ` for a lattice location or `NewAgent` for a continuous position. Move it with `MoveSQ` or `Move`, and remove it with `Dispose`.

`agents.All()` returns a snapshot of living handles. That allows creation and disposal during an iteration without changing the iteration's current membership. On a nonstackable grid, safe mode rejects attempts to place two agents at the same site. Use `isStackable=True` only when multiple occupancy is part of the model.

## 5. Simultaneous population and field updates

Both `PopGrid` and `PDEgrid` distinguish current state from pending changes. Direct indexing changes current state immediately; `Add` queues a delta, and `Update` applies queued changes. `Reset` clears both current state and pending changes.

A `PDEgrid` also provides diffusion and advection. Set the timestep and spatial spacings with `SetTimeSpaceStep(dt, dx, dy, dz)` before transport operations. Cartesian, interface-based, masked, ADI, and radial diffusion variants are available; choose the method that matches the model's geometry and numerical assumptions. Validate timestep stability and boundary behavior in safe mode before optimizing.

## 6. Compiled model code

Decorate computational functions with `@pal.njit`. PAL expands some model constructs before passing the function to Numba, including neighborhood iteration and source-aware safe-mode diagnostics. Annotate PAL arguments with their public types, such as `pal.AgentGrid` or `pal.PDEgrid`, for clarity and compiler support.

Keep model state in PAL objects and use `pal.Seed`, `pal.Random`, `pal.RandInt`, and `NewMultinomial` for reproducible PAL random sampling. Run a model in safe mode first. Fast mode removes selected checks; it does not make invalid indices, dead handles, or unstable numerical steps valid.

## 7. Drawing and output

`StartPixWindow` creates a pixel grid and a window for lattice visualization. Assign packed RGB colors such as `0xFF0000` to pixels and call `window.Update()` to publish the frame. `StartOpenGLWindow` provides geometric 2D/3D drawing. Windows can save images and GIFs; `headless=True` supports off-screen workflows when the required rendering backend is installed.

Rendering and simulation are separate. A model can run without a window, draw every timestep, or draw only selected checkpoints.

## 8. How PAL works

PAL separates its public Python interface from a native C core. Python constructors create native-backed model state. Thin Numba-compatible wrappers allow compiled model code to access that state without repeatedly returning to Python. The public Protocol classes describe the annotation/autocomplete surface while safe and fast concrete implementations provide runtime behavior.

This design favors fast repeated timesteps, but the first compiled call may include substantial Numba compilation time. Measure **cold compilation** separately from **steady-state execution**. Safe mode includes additional validation and source-aware diagnostics; fast mode removes some of that overhead. A change that improves one benchmark may regress another, so measure both modes on representative workloads.

## 9. Where to go next

Read the [API Guide](API_GUIDE.md) for the complete public subsystem map. Consult the [API Reference](API_REFERENCE.md) for current declarations and the repository's `Examples/` for larger models. Use `Testing/` for executable contract examples, especially around boundaries, wrapping, pending updates, and invalid operations.
