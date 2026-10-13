# PAL Cheatsheet

`import PythonAutomataLibrary as pal` · Safe mode is default. Call `pal.FastMode()` before constructing any PAL object. Put substantive model kernels under **`@pal.njit(cache=True)`**.

Axis shorthand: `...` after an X/x-specific API name denotes corresponding Y/y and Z/z variants.

Curated one-page API map; see API Guide and API Reference for details.

## START / UTILITIES

- `pal.FastMode() · pal.njit(cache=True)`: Select fast mode before object creation; compile model kernels.
- `pal.Seed(s) · Random() · RandInt(n)`: Shared PAL RNG; seed for reproducibility.
- `pal.NewGrid(dims,dtype) · NewAgentGrid(dims,props=0,stack=False)`: Typed sites; mobile agents.
- `pal.NewPopGrid(dims,capacity=None) · NewPDEgrid(dims)`: Integer populations; continuous fields.
- `pal.NewIList() · NewMultinomial(other=None)`: Reusable integer list; random sampler.
- `pal.MooreHood(dim,excludeCenter=False) · VonNeumannHood(...) · CircleHood(dim,rad,...)`: Relative neighbor offsets.
- `pal.RGB(r,g,b) · GetR/G/B(c) · ColorScale(colors,value)`: Pack, extract, interpolate colors.
- `pal.AwaitWindows(timeout=30)`: Wait for rendering windows.

## SHARED GRID API

- `dims=(nx,ny,nz); negative size = wrap`: 1–3 dimensions; unused axes omitted.
- `grid.xDim... · nDims · wrapX...`: Axis sizes, dimensionality, periodicity.
- `len(grid) · grid[i] · grid[x,y,z] · grid[...] = value`: Site count; read/write; slices copy.
- `grid.ToI(x,y,z) · ItoX...(i)`: Coordinate ↔ linear index.
- `grid.InWrapX...(x)`: Wrap coordinate on periodic axis.
- `grid.Box(x1,x2,y1,y2,z1,z2)`: Half-open rectangular iterator.
- `grid.Hood(hood,x,y,z,unroll=False)`: Offset neighborhood at a site.

## AGENT GRID

- `a=agents.NewAgentSQ(x,y,z) · NewAgent(x,y,z)`: Create at lattice site / continuous point.
- `agents.MoveSQ(a,x,y,z) · Move(a,x,y,z)`: Move by lattice / continuous position.
- `agents.I(a) · XSQ...(a) · X...(a)`: Site index, lattice and real coordinates.
- `agents.Alive(a) · Dispose(a) · GetPop()`: Life status, removal, count.
- `agents[a,p] · agents.counts`: Agent properties; read-only occupancy.
- `agents.AgentsAt(x,y,z) · LastAgent(x,y,z)`: Site occupants / last occupant.
- `agents.AgentsInRadius(rad,x,y,z,exclude=None)`: Nearby agents; periodic-aware.
- `agents.All(shuffle=False)`: Snapshot of living agent IDs.
- `agents.InWrapSQX...(x) · DispWrapX...(p1,p2)`: Wrap lattice coordinate; shortest displacement.

## POP GRID / PDE FIELD

- `pop.Add(v,x,y,z) · Update() · Reset()`: Queue count deltas, apply, clear.
- `pop.GetPop() · All()`: Total population; occupied indices.
- `field.Add(v,x,y,z) · Update() · Reset()`: Queue field deltas, apply, clear.
- `field.SetTimeSpaceStep(dt,dx,dy=1,dz=1) · Dt() · Dx...()`: Numerical spacing.
- `field.Diffusion(rate,BC...) · DiffusionMask(rate,mask,...)`: Uniform / masked diffusion.
- `field.DiffusionField(rates,BC...) · DiffusionInterfaces(rx,ry,rz,BC...)`: Cell / face coefficients.
- `field.DiffusionADI(rate,BC...) · DiffusionRadialCircle/Sphere(rate,outerBC)`: Implicit / radial diffusion.
- `field.Advection(vx,vy,vz,BC...) · AdvectionField(vx,vy,vz,BC...)`: Uniform / spatial velocity.
- `field.AdvectionInterfaces(vx,vy,vz,BC...)`: Face-centered velocity.
- `BC... = xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC`: Optional face boundary conditions.

## LISTS / SAMPLING

- `lst.Append(i) · Clear() · Random() · Shuffle()`: Integer-list operations.
- `lst.All() · Iter() · len(lst) · lst[i]`: Copy / no-copy traversal, indexing.
- `m.Binomial(n,p) · Setup(n) · Sample(p)`: Binomial sampling.

## SHARED WINDOW API

- `win.IsOpen() · Update() · Save(path,block=False) · Close()`: Window lifecycle / image output.
- `win.StartGif(path,delay=100) · AddGifFrame(block=False) · StopGif()`: GIF recording; OpenGL supports timeout options.

## PIX WINDOW

- `pix,win=pal.StartPixWindow(nx,ny,scale=1,title='PAL',headless=False)`: Pixel renderer.
- `pix[x,y]=pal.RGB(r,g,b) · pix[i]=color`: Draw packed RGB pixels.
- `pix.xDim · yDim · ToI(x,y) · ItoX/Y(i) · len(pix)`: Buffer geometry.

## OPENGL WINDOW / DRAW

- `draw,win=pal.StartOpenGLWindow(nx,ny,nz=None,width=800,height=800,...)`: 2D/3D primitive renderer.
- `draw.Circle(rad,color,x,y,z=0) · Box(xLen,color,x,y,z=0,...)`: Add circles/spheres and boxes.
- `draw.BoxSQ(color,x,y,z=0) · Line(width,color,x1,y1,x2,y2,...)`: Site boxes and segments.
- `draw.Clear() · win.Clear() · win.Background(color)`: Reset primitives / scene / background.
- `win.Borders(width,color) · win.Camera(x,y,z,yaw=None,pitch=None)`: Domain border and 3D camera.
