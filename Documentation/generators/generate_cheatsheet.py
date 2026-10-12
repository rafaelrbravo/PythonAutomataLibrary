"""Generate PAL CHEATSHEET.md and CHEATSHEET.pdf from one compact definition.

The PDF uses a readable two-column, full-height single-page landscape layout generated from the same
structured content as the Markdown cheatsheet.

Usage: python Documentation/generators/generate_cheatsheet.py [--check]
"""
import argparse, html, re
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph

ROOT = Path(__file__).resolve().parents[2]
MD_TARGET = ROOT / "Documentation" / "CHEATSHEET.md"
PDF_TARGET = ROOT / "Documentation" / "CHEATSHEET.pdf"
TITLE = "Python Automata Library (PAL) Cheat Sheet"
INTRO = "`import PythonAutomataLibrary as pal` · Safe mode is default. Call `pal.FastMode()` before constructing any PAL object. Put substantive model kernels under **`@pal.njit(cache=True)`**."
CONVENTIONS = "`dims` gives grid size per axis: `(nx,)` for 1D, `(nx, ny)` for 2D, `(nx, ny, nz)` for 3D. Negative dimensions enable wrapping; out-of-range coordinates wrap on wrapped axes and are skipped on nonwrapped axes. `x...` means `x`, `x,y`, or `x,y,z` in 1D/2D/3D."

SECTIONS = [
("Create state", [
("`pal.NewGrid(dims, dtype)`", "Create one value per site; supports bool, fixed-width integer, and float dtypes."), ("`pal.NewAgentGrid(dims, numAgentProps=0, isStackable=False)`", "Create mobile agents with per-agent properties. `isStackable` permits multiple agents per site; `dims=()` makes a nonspatial collection."),
("`pal.NewPopGrid(dims, capacity=None)`", "Store integer population counts per site; `capacity` limits total population."), ("`pal.NewPDEgrid(dims)`", "Create a continuous field with queued synchronous updates and diffusion/advection operators."),
("`pal.NewIList()`", "Create a mutable integer list for collecting/reusing query results."), ("`pal.NewMultinomial()`", "Create a sampler for binomial/multinomial population draws."),]),
("Shared lattice geometry", [
("Properties", "`xDim` / `yDim` / `zDim`: axis sizes; `nDims`: number of axes; `wrapX` / `wrapY` / `wrapZ`: wrapping flags."),
("`ToI(x...)`", "Convert coordinates to a linear site index; inverse methods are `ItoX/Y/Z(i)`."),


("`grid.Hood(hood, x...)`", "Apply relative coordinate offsets, e.g. `((x1,y1),(x2,y2),...)`, at `x...`. Special-case hoods: **`pal.MooreHood`**`(*)`, **`pal.VonNeumannHood`**`(*)`, **`pal.CircleHood`**`(*,rad,*)`. `*` represents shared arguments: `dim` (number of dimensions) and `excludeCenter=False` (omit the center when true)."),]),
("Grid / common indexing", [
("Indexing", "`grid[i]` reads/writes by linear index; `grid[x...]` uses dimensional coordinates."), ("Slices", "Return detached NumPy copies, not views; write back explicitly to change the grid."),
("`Box(x1,x2,...)`", "Select a rectangular region: `(xLow,xHigh,yLow,yHigh,...)`, with exclusive upper bounds."),
]),
("Draw / output", [
("`pal.StartPixWindow(...)`", "Create a pixel buffer and display window; `headless=True` supports noninteractive rendering."),
("`pix[x,y] = RGB`", "Set a pixel color in the shared pixel buffer."),
("Color utilities", "`pal.RGB(r,g,b)` packs red, green, and blue channels into a color; `pal.GetR(color)`, `pal.GetG(color)`, and `pal.GetB(color)` extract channels; `pal.ColorScale(...)` maps values to colors."),
("PixWindow controls", "`win.Update()` refreshes the display; `win.Save(path,block=True)` saves an image; `win.Close()` releases the window."),
("PixWindow GIF", "`win.StartGif(path,delay=100)` starts recording; `win.AddGifFrame(block=False)` appends a frame; `win.StopGif()` finishes."),
("`pal.StartOpenGLWindow(xDim,yDim,zDim=None,width=800,height=800,...)`", "Create the renderer and return `(draw, win)`: `draw` collects primitives; `win` handles display, updates, and output. Omit `zDim` for 2D; set it for 3D. `headless=True` supports noninteractive rendering."),
("`draw.Circle(rad,color,x,y,z=None)`", "Add a circle or sphere-like primitive with the given radius and color."),
("`draw.Box(xLen,color,x,y,z=None,yLen=None,zLen=None)`", "Add a box centered at a position; specify axis lengths to control its size."),
("`draw.BoxSQ(color,x,y,z=None)`", "Add a box aligned to a lattice site."),
("`draw.Line(width,color,x1,y1,x2,y2,z1=None,z2=None)`", "Add a line segment between two 2D/3D points."),
("`draw.Borders(width,color)`", "Draw grid borders with the requested line width and color."),
("`win.Camera(x,y,z,yaw=None,pitch=None)`", "Set the camera position and optional yaw/pitch angles for a 3D scene."),
("OpenGL frame", "`win.Background(color)` sets the background; `draw.Clear()` clears collected primitives; `win.Clear()` clears the scene; `win.Update()` renders."),
("OpenGL output", "`win.IsOpen()` checks the window; `win.Save(path,block=False)` saves an image; `win.Close()` releases resources."),
("OpenGL GIF", "`win.StartGif(...)` begins recording; `win.AddGifFrame(...)` adds a frame; `win.StopGif(...)` finishes."),]),("Lists + randomness", [
("IList methods", "`Append(i)` adds an integer; `Clear()` empties the list; `Random()` picks an entry; `Shuffle()` reorders entries; `All()` copies the list; `Iter()` traverses without copying."),
("RNG methods", "`pal.Seed(seed)` seeds PAL’s RNG; `pal.Random()` draws uniform [0,1); `pal.RandInt(n)` draws an integer in [0,n). PAL’s RNG is separate from NumPy’s."),
("Multinomial methods", "`m.Binomial(n,p)` draws a binomial count; `m.Setup(...)` configures multinomial probabilities; `m.Sample(...)` draws counts."),]),
("AgentGrid", [
("`NewAgentSQ(x...)` / `MoveSQ(a,x...)`", "Create/move agents by integer site coordinates; linear-site variants are also available."),
("`NewAgent(x...)` / `Move(a,x...)`", "Create/move agents by continuous coordinates (off-lattice)."),
("`grid[a,p]`", "Read or write property p for agent a."),
("Agent position", "`I(a)` gives the linear site index; `XSQ/YSQ/ZSQ(a)` give lattice coordinates; `X/Y/Z(a)` give continuous coordinates."),
("`Alive(a)` / `Dispose(a)`", "Check whether an agent is still valid, or remove it from the grid."),
("`GetPop()`", "Count agents currently in the grid."),
("`AgentsAt(x...)` / `LastAgent(x...)`", "Query all agents at a site or just its last agent; useful for stackable grids."),
("`AgentsInRadius(...)`", "Find nearby agents, respecting periodic boundaries."),
("`counts[x...]`", "Read site occupancy counts without modifying them."),
("Iteration", "`agents.All()` returns a detached snapshot, safe for iteration while adding or disposing agents."),
("`DispWrapX/Y/Z(p1,p2)`", "Compute the shortest displacement along a periodic axis."),]),
("PopGrid + PDEgrid", [
("Synchronous updates", "`Add(v,...)` queues a change; `Update()` applies queued changes together; `Reset()` clears current and pending values. Indexed assignment changes current state immediately."),
("`pop.GetPop()`", "Return total population across all sites."),
("`pop.All()`", "Return occupied site indices (not their counts)."),
("PDE setup", "`field.SetTimeSpaceStep(dt, dx[,dy,dz])` sets time and axis spacing; explicit updates require stable time steps."),
("Diffusion methods", "`Diffusion(...)` uses the standard scheme; `DiffusionMask(...)` applies a mask; `DiffusionField(...)` uses spatially varying coefficients; `DiffusionInterfaces(...)` specifies interfaces; `DiffusionADI(...)` uses an implicit scheme; `DiffusionRadialCircle/Sphere(...)` handles radial geometry."),
("Advection methods", "`Advection(...)` uses constant velocity; `AdvectionField(...)` uses spatially varying velocity; `AdvectionInterfaces(...)` specifies interface velocities."),]),

]

FOOTER = "More detail: Manual · API Guide · API Reference (Documentation/)."


def _md():
    out=["# PAL Cheatsheet", "", "## Conventions", "", INTRO + " " + CONVENTIONS, ""]
    for title, entries in SECTIONS:
        out += [f"## {title}", ""]
        for key, desc in entries: out.append(f"- {_md_markup(key, heading=True)}: {_md_markup(desc, heading=(key in ('Properties', 'Indexing')))}")
        out.append("")
    out.append(f"**{FOOTER}**")
    return "\n".join(out).rstrip()+"\n"


def _md_markup(s, heading=False):
    s=re.sub(r"`([A-Za-z_][A-Za-z_0-9.]*)\(([^`]*)\)`", r"`**\1**(\2)`", s)
    if heading:
        s=re.sub(r"`([^`]+)`", lambda m: m.group(0) if "**" in m.group(1) else "`**" + m.group(1) + "**`", s)
    return s


def _markup(s, heading=False):
    s=html.escape(s, quote=False).replace("-&gt;", "→")
    s=re.sub(r"\*\*`([^`]+)`\*\*", r"<b>\1</b>", s)
    s=re.sub(r"`([A-Za-z_][A-Za-z_0-9.]*)\(([^`]*)\)`", r"<b>\1</b>(\2)", s)
    if heading:
        s=re.sub(r"`([^`]+)`", r"<b>\1</b>", s)
    else:
        s=re.sub(r"`([^`]+)`", r"\1", s)
    return s


def _build_pdf(body_font=8.0):
    """Render a deliberately selective one-page quick-reference; Markdown retains all entries."""
    buf = BytesIO()
    pw, ph = landscape(letter)
    margin = .30 * inch
    header = .32 * inch
    gutter = .22 * inch
    col_w = (pw - 2 * margin - gutter) / 2
    usable = ph - 2 * margin - header
    frames = [Frame(margin + i * (col_w + gutter), margin, col_w, usable,
                    leftPadding=3, rightPadding=3, topPadding=1, bottomPadding=1)
              for i in range(2)]

    def header_fn(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(margin, ph - margin - 8, TITLE)
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(pw - margin, ph - margin - 7, "Full details: API Guide · API Reference")
        canvas.setStrokeColor(colors.HexColor("#888888"))
        canvas.line(margin, ph - margin - 13, pw - margin, ph - margin - 13)
        canvas.restoreState()

    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11,
                             leading=12.5, spaceBefore=8, spaceAfter=4,
                             backColor=colors.HexColor("#F1F1F1"), borderPadding=2)
    entry = ParagraphStyle("entry", fontName="Helvetica", fontSize=body_font,
                           leading=body_font * 1.22, spaceAfter=3.2,
                           leftIndent=5, firstLineIndent=-5)
    intro = ParagraphStyle("intro", fontName="Helvetica", fontSize=body_font,
                           leading=body_font * 1.22, spaceAfter=6)

    # The printable sheet is intentionally selective. The Markdown is exhaustive.
    # Use short descriptions instead of compressing the complete prose into tiny type.
    selected = [
        ("Create state", [
            ("NewGrid", "pal.NewGrid(dims,dtype): values at lattice sites"),
            ("NewAgentGrid", "pal.NewAgentGrid(dims,numAgentProps=0,isStackable=False): individual agents"),
            ("NewPopGrid", "pal.NewPopGrid(dims,capacity=None): integer counts"),
            ("NewPDEgrid", "pal.NewPDEgrid(dims): continuous field"),
            ("NewIList", "pal.NewIList(): mutable integer list"),
            ("NewMultinomial", "pal.NewMultinomial(): binomial/multinomial draws"),
        ]),
        ("Geometry / indexing", [
            ("Dimensions", "dims=(nx,), (nx,ny), or (nx,ny,nz); negative sizes wrap"),
            ("Coordinates", "grid[i] linear index; grid[x,y,z] site coordinates"),
            ("Conversions", "ToI(x,...); ItoX/Y/Z(i)"),
            ("Neighbors", "Hood(hood,x,...); pal.MooreHood, VonNeumannHood, CircleHood"),
            ("Region", "Box(...) iterates a rectangular coordinate region"),
        ]),
        ("AgentGrid", [
            ("Birth", "NewAgentSQ(x,...) lattice; NewAgent(x,...) continuous"),
            ("Move", "MoveSQ(a,x,...) lattice; Move(a,x,...) continuous"),
            ("Properties", "agents[a,p] reads/writes property p"),
            ("Position", "I(a); XSQ/YSQ/ZSQ(a); X/Y/Z(a)"),
            ("Lifetime", "Alive(a); Dispose(a); GetPop()"),
            ("Queries", "AgentsAt(...); LastAgent(...); AgentsInRadius(...)"),
            ("Iteration", "All(shuffle=True) snapshots living agent handles"),
        ]),
        ("Population / fields", [
            ("State", "grid[i] = v writes immediately"),
            ("Queued change", "Add(v,...) queues; Update() applies; Reset() clears"),
            ("Population", "pop.GetPop() total; pop.All() occupied indices"),
            ("PDE spacing", "field.SetTimeSpaceStep(dt,dx,dy,dz)"),
            ("Diffusion", "Diffusion, DiffusionMask, DiffusionField, DiffusionInterfaces, DiffusionADI"),
            ("Radial", "DiffusionRadialCircle / DiffusionRadialSphere"),
            ("Advection", "Advection, AdvectionField, AdvectionInterfaces"),
        ]),
        ("Random / lists", [
            ("RNG", "pal.Seed(seed); pal.Random(); pal.RandInt(n)"),
            ("IList", "Append, Clear, Random, Shuffle, All (copy), Iter (no copy)"),
            ("Sampling", "m.Binomial(n,p); m.Setup(...); m.Sample(...)"),
        ]),
        ("Drawing / execution", [
            ("Compile", "@pal.njit(cache=True) for model kernels"),
            ("Mode", "Safe by default; pal.FastMode() before object creation"),
            ("Pixels", "pal.StartPixWindow(...); pix[x,y] = color; win.Update()"),
            ("OpenGL", "pal.StartOpenGLWindow(...); draw.Circle/Box/Line(...)"),
            ("Output", "win.Save(path); StartGif, AddGifFrame, StopGif; Close"),
        ]),
    ]
    story = [Paragraph(_markup(INTRO), intro)]
    for section_index, (title, rows) in enumerate(selected):
        # Three sections per column; balance both columns across the full sheet.
        if section_index == 3:
            story.append(FrameBreak())
        story.append(Paragraph(title, heading))
        for key, desc in rows:
            story.append(Paragraph("<b>" + html.escape(key) + "</b>: " + html.escape(desc), entry))
    doc = BaseDocTemplate(buf, pagesize=landscape(letter), leftMargin=margin,
                          rightMargin=margin, topMargin=margin, bottomMargin=margin,
                          invariant=1)
    doc.addPageTemplates(PageTemplate(id="main", frames=frames, onPage=header_fn))
    doc.build(story)
    return buf.getvalue(), doc.page


def _pdf():
    data, pages = _build_pdf(10.0)
    if pages != 1:
        raise RuntimeError(f"Cheatsheet must be exactly one page; generated {pages}.")
    print("PAL cheatsheet generated 1 landscape page, 2 full-height columns at 10 pt body font.")
    return data

def build(): return _md(), _pdf()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--check",action="store_true"); a=p.parse_args(); md,pdf=build()
    if a.check:
        stale=(not MD_TARGET.exists() or MD_TARGET.read_text(encoding="utf-8")!=md or not PDF_TARGET.exists() or PDF_TARGET.read_bytes()!=pdf)
        if stale: p.exit(1,"Stale or missing Cheatsheet artifacts; run Documentation/generators/generate_cheatsheet.py\n")
        print("CHEATSHEET.md and CHEATSHEET.pdf are current")
    else:
        MD_TARGET.write_text(md,encoding="utf-8"); PDF_TARGET.write_bytes(pdf); print(f"Wrote {MD_TARGET.relative_to(ROOT)} and {PDF_TARGET.relative_to(ROOT)}")
if __name__=="__main__": main()
