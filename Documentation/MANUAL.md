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

The example contains 1,600 sites and runs 10 increments per site, so the final total population is 16,000. `Add` accumulates changes; `Update` applies them together. That distinction is useful when one site's update must not affect its neighbors during the same timestep. The printed value is `16000`; every site contains `10`.

Begin in PAL's default **safe mode**. Only select `pal.FastMode()` before constructing any PAL object, after the model is validated and when performance measurements justify it.

## 2. Choose the right representation

A `Grid` holds one typed value per site. An `AgentGrid` holds individually identifiable agents, optionally with per-agent properties. A `PopGrid` holds integer counts without individual identity. A `PDEgrid` holds a continuous scalar field and transport operations. These can be combined: for example, agents consume a PDE field while population counts accumulate in a separate PopGrid.

Use a tuple of dimensions when constructing a spatial grid. Negative dimensions mean periodic wrapping on that axis, so `(-40, 40)` wraps x but not y. `AgentGrid` also supports an empty dimension tuple for a nonspatial population.

### Choosing a representation by scale

Individual agents are useful when identity, history, or per-agent properties affect behavior. If agents are interchangeable within a site, a `PopGrid` replaces many handles with one integer count per site and can be much cheaper to update. A `Grid` is appropriate for a site attribute that is not a population, such as a terrain type or a fixed mask. Use `PDEgrid` for a field whose values change continuously through diffusion or advection. A hybrid model can use different representations for different scales, but must explicitly define how state is exchanged between them.

Choose a representation based on what the model must preserve, not just on the number of sites. Converting individual agents to counts loses individual identity and history; representing a count as a continuous field can lose integer and stochastic behavior. Those are modeling assumptions, not implementation details.

## 3. Coordinates and neighborhoods

A grid accepts linear lattice indices. In a 2D grid, `ToI(x, y)` converts coordinates to an index, and `ItoX(i)` and `ItoY(i)` convert back. Grid geometry is exposed as attributes such as `xDim`, `yDim`, `nDims`, and `wrapX`, not methods.

PAL provides `MooreHood`, `VonNeumannHood`, and `CircleHood`. They return relative integer offsets. `grid.Hood(hood, x, y)` maps a neighborhood around a site, taking the grid's periodic boundaries into account. `grid.Box(...)` iterates a half-open rectangular region.

## 4. Individual agents

An agent is an integer handle owned by an `AgentGrid`. Create it with `NewAgentSQ` for a lattice location or `NewAgent` for a continuous position. Move it with `MoveSQ` or `Move`, and remove it with `Dispose`.

`agents.All()` returns a snapshot of living handles. That allows creation and disposal during an iteration without changing the iteration's current membership. On a nonstackable grid, safe mode rejects attempts to place two agents at the same site. Use `isStackable=True` only when multiple occupancy is part of the model.

### Iterating nearby agents

Use `AgentsAt(x, y)` to visit the agents occupying one lattice site. For a continuous search around a point, `AgentsInRadius(rad, x, y)` yields `(agent, dx, dy, distSq)` in 2D. The displacement components account for periodic wrapping and `distSq` is squared distance. In 1D the tuple is `(agent, dx)`; in 3D it is `(agent, dx, dy, dz, distSq)`. Pass `exclude=agent` to omit an agent from its own neighborhood search.

Both iterators work in Python and inside `@pal.njit` loops. Python iteration materializes matching handles first; this does not permit structural modification of the grid during iteration. Safe mode checks for structural changes while iterating, whereas fast mode omits the check. For a loop that creates or disposes agents, iterate a snapshot from `agents.All()` instead.

For example, this Python-side search finds the agent across a periodic x boundary:

```python
import PythonAutomataLibrary as pal

agents = pal.NewAgentGrid((-10, 10))
agent = agents.NewAgent(9.0, 2.0)
nearby = list(agents.AgentsInRadius(1.5, 0.0, 2.0))
assert len(nearby) == 1
found, dx, dy, distSq = nearby[0]
assert found == agent
assert dx == -1.0 and dy == 0.0 and distSq == 1.0
assert list(agents.AgentsInRadius(1.5, 0.0, 2.0, exclude=agent)) == []
```

## 5. Simultaneous population and field updates

Both `PopGrid` and `PDEgrid` distinguish current state from pending changes. Direct indexing changes current state immediately; `Add` queues a delta, and `Update` applies queued changes. `Reset` clears both current state and pending changes.

A `PDEgrid` also provides diffusion and advection. Set the timestep and spatial spacings with `SetTimeSpaceStep(dt, dx, dy, dz)` before transport operations. Cartesian, interface-based, masked, ADI, and radial diffusion variants are available; choose the method that matches the model's geometry and numerical assumptions. Validate timestep stability and boundary behavior in safe mode before optimizing.

### Numerical update order

The timestep order is part of the model. For example, queue all births and deaths in a `PopGrid` before calling `Update()` if changes must be simultaneous. Reading current counts while queuing deltas then uses the same starting population throughout the step. By contrast, assigning directly to `pop[x, y]` changes the value that subsequent calculations read. Mixing immediate assignments and queued changes is possible, but their order then changes the model.

For a `PDEgrid`, transport routines accumulate changes that become current at `Update()`. Check timestep stability and convergence by varying the timestep and grid spacing. Safe-mode checks do not replace numerical validation.

Consider two sites with populations 10 and 0. Suppose each site sends half its starting population to the other site in one timestep. With queued deltas, site 0 queues -5 and site 1 queues +5, then a single `Update()` gives (5, 5). If the first transfer were assigned immediately and the second site were subsequently processed using its new count, the second calculation would see 5 rather than the original 0. This is an algorithmic difference, not a rounding issue. The same distinction applies when several agents consume a shared resource or when births and deaths are calculated from local densities.

For a transfer that depends on the initial state, compute both changes before applying either one. `PopGrid.Add(delta, i)` records each site's signed change, and `Update()` commits all recorded changes together. This preserves the intended conservation law: the sum of the two deltas is zero, so the total population remains 10. Conservation is a useful test for transport or movement routines; it catches accidental creation or loss even when individual site values look plausible.

When debugging a population model, test these properties separately: site counts remain nonnegative when the model requires it; the total changes only through explicit births, deaths, or boundary flux; and results do not depend on the order in which sites are visited when an update is meant to be simultaneous. Compare one small hand-calculated step against the implementation before running a large simulation.

## 6. Compiled model code

Decorate computational functions with `@pal.njit`. PAL expands some model constructs before passing the function to Numba, including neighborhood iteration and source-aware safe-mode diagnostics. Annotate PAL arguments with their public types, such as `pal.AgentGrid` or `pal.PDEgrid`, for clarity and compiler support.

Keep model state in PAL objects and use `pal.Seed`, `pal.Random`, `pal.RandInt`, and `NewMultinomial` for reproducible PAL random sampling. Run a model in safe mode first. Fast mode removes selected checks; it does not make invalid indices, dead handles, or unstable numerical steps valid.

## 7. Drawing and output

`StartPixWindow(xDim, yDim, scale=1, title='PAL', headless=False)` returns `(pix, window)`. Write packed RGB colors such as `0xFF0000` to `pix[x, y]`, then call `window.Update()` to publish the frame. `window.Save(path, block=True)` waits for the image to be written; use `window.Close()` to release resources. The pixel buffer also supports linear indices, slices, and writes from compiled code. For off-screen images, use `headless=True` (requires the relevant output dependencies). GIF recording uses `window.StartGif(path, delay=100)`, `window.AddGifFrame(block=False)`, and `window.StopGif()`; update the window before capturing each frame.

`StartOpenGLWindow` provides geometric 2D/3D drawing with `Circle`, `Box`, `BoxSQ`, `Line`, `Borders`, `Camera`, `Background`, and `Clear`. It also supports image/GIF output and `headless=True`, but needs a working OpenGL backend. Close windows after use.

Rendering and simulation are separate. A model can run without a window, draw every timestep, or draw only selected checkpoints.

## 8. How PAL works

PAL separates its public Python interface from a native C core. Python constructors create native-backed model state. Thin Numba-compatible wrappers allow compiled model code to access that state without repeatedly returning to Python. The public Protocol classes describe the annotation/autocomplete surface while safe and fast concrete implementations provide runtime behavior.

This design favors fast repeated timesteps, but the first compiled call may include substantial Numba compilation time. Measure **cold compilation** separately from **steady-state execution**. Safe mode includes additional validation and source-aware diagnostics; fast mode removes some of that overhead. A change that improves one benchmark may regress another, so measure both modes on representative workloads.

## 9. Checkpointing and reproducibility

PAL's native-backed model objects support Python pickling, so a long-running model can save its state and resume without retaining native pointers from the original process. A checkpoint must include **all** state needed for continuation: every grid, relevant parameters, the current timestep, and any additional state managed by the model. Saving only one grid does not preserve a coupled simulation. Keep the code version and parameter configuration alongside the checkpoint so that results can be interpreted later.

For reproducibility, seed the random generator before a run and record the seed. Verify that a restored model continues from the expected state; don't assume that pickling a grid also captures unrelated Python variables or the state of every random-number generator used by external libraries. Compare a short uninterrupted run with a checkpoint-and-resume run as a regression test.

A minimal checkpoint can be saved and resumed without opening a window:

```python
import pickle
import PythonAutomataLibrary as pal

pop = pal.NewPopGrid((5,))
pop[1] = 10
pop.Add(7, 1)  # Pending change is part of the checkpoint.
state = {"pop": pop, "step": 3}
checkpoint = pickle.dumps(state)

restored = pickle.loads(checkpoint)
restored["pop"].Update()
assert restored["step"] == 3
assert restored["pop"][1] == 17
```

The repository includes a working example at `Examples/Agents/SaveLoadModel.py`. It serializes a dictionary containing an `AgentGrid`, an `IList`, and the timestep, then restores all three before continuing. The regression suite `Testing/test_13_pickle_state.py` also verifies that `PopGrid` and `PDEgrid` retain pending deltas across a pickle round-trip and that restored agent grids preserve living handles and properties. These tests establish PAL object-state restoration, not restoration of external RNG streams.

## 10. Where to go next

Read the [API Guide](API_GUIDE.md) for the complete public subsystem map. Consult the [API Reference](API_REFERENCE.md) for current declarations and the repository's `Examples/` for larger models. Use `Testing/` for executable contract examples, especially around boundaries, wrapping, pending updates, and invalid operations.
