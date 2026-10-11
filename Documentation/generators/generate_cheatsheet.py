"""Generate PAL CHEATSHEET.md and CHEATSHEET.pdf from one compact definition.

The PDF uses a readable two-column, multi-page layout generated from the same
structured content as the Markdown cheatsheet.

Usage: python Documentation/generators/generate_cheatsheet.py [--check]
"""
import argparse, html, re
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph

ROOT = Path(__file__).resolve().parents[2]
MD_TARGET = ROOT / "Documentation" / "CHEATSHEET.md"
PDF_TARGET = ROOT / "Documentation" / "CHEATSHEET.pdf"
TITLE = "Python Automata Library (PAL) Cheat Sheet"
INTRO = "`import PythonAutomataLibrary as pal` · Safe mode is default. Call `pal.FastMode()` before constructing any PAL object. Put substantive model kernels under **`@pal.njit(cache=True)`**."
CONVENTIONS = "`dims` gives grid size per axis: `(nx,)` for 1D, `(nx, ny)` for 2D, `(nx, ny, nz)` for 3D. Negative dimensions enable wrapping; out-of-range coordinates wrap on wrapped axes and are skipped on nonwrapped axes. `x...` means `x`, `x,y`, or `x,y,z` in 1D/2D/3D."

SECTIONS = [
("Create state", [
("`pal.NewGrid(dims, dtype)`", "Create a typed lattice storing one value per site."), ("`pal.NewAgentGrid(dims, numAgentProps=0, isStackable=False)`", "Create agents with optional properties; use `dims=()` for nonspatial agents."),
("`pal.NewPopGrid(dims, capacity=None)`", "Create integer counts per site; capacity limits total population."), ("`pal.NewPDEgrid(dims)`", "Create a continuous field for PDE updates."),
("`pal.NewIList()`", "Create a mutable integer list for collecting/reusing query results."), ("`pal.NewMultinomial()`", "Create a sampler for binomial/multinomial population draws."),]),
("Shared lattice geometry", [
("Properties", "`xDim` / `yDim` / `zDim`: axis sizes; `nDims`: number of axes; `wrapX` / `wrapY` / `wrapZ`: wrapping flags."),
("`ToI(x...)`", "Convert 1D/2D/3D coordinates into one linear site index."),
("`ItoX/Y/Z(i)`", "Convert a linear site index back to its axis coordinate."),

("`grid.Hood(hood, x...)`", "Apply relative coordinate offsets, e.g. `((x1,y1),(x2,y2),...)`, at `x...`. Special-case hoods: `pal.MooreHood`, `pal.VonNeumannHood`, `pal.CircleHood`. All accept `dim` (number of dimensions) and `excludeCenter=False` (whether to omit the center)."),]),
("Grid / common indexing", [
("Indexing", "`grid[i]` reads/writes by linear index; `grid[x...]` uses dimensional coordinates."), ("Slices", "Return detached NumPy copies, not live views; editing a slice does not update the grid."),
("`Box(x1,x2,...)`", "Select a rectangular region with lower and exclusive upper bounds for each axis, in axis order."),
]),
("Draw / output", [
("`pal.StartPixWindow(...)`", "Create a pixel buffer and display window; `headless=True` supports noninteractive rendering."),
("`pix[x,y] = RGB`", "Set a pixel color in the shared pixel buffer."),
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
("`NewAgentSQ(x...)` / `MoveSQ(a,x...)`", "Create or move an agent using lattice-site coordinates; documented linear-site forms are also supported."),
("`NewAgent(x...)` / `Move(a,x...)`", "Create or move an agent using continuous coordinates rather than lattice sites."),
("`grid[a,p]`", "Read or write property p for agent a."),
("Agent position", "`I(a)` gives the linear site index; `XSQ/YSQ/ZSQ(a)` give lattice coordinates; `X/Y/Z(a)` give continuous coordinates."),
("`Alive(a)` / `Dispose(a)`", "Check whether an agent is still valid, or remove it from the grid."),
("`GetPop()`", "Count agents currently in the grid."),
("`AgentsAt(x...)` / `LastAgent(x...)`", "Find agents at a site or retrieve the last agent there."),
("`AgentsInRadius(...)`", "Find nearby agents, respecting periodic boundaries."),
("`counts[x...]`", "Read site occupancy counts without modifying them."),
("Iteration", "`agents.All()` returns a detached snapshot, safe for iteration while adding or disposing agents."),
("`DispWrapX/Y/Z(p1,p2)`", "Compute the shortest displacement along a periodic axis."),]),
("PopGrid + PDEgrid", [
("Synchronous updates", "`Add(v,...)` queues a change; `Update()` applies queued changes together; `Reset()` clears current and pending values. Indexed assignment changes current state immediately."),
("`pop.GetPop()`", "Return total population across all sites."),
("`pop.All()`", "Return indices of sites with nonzero population, not the population values."),
("PDE setup", "`field.SetTimeSpaceStep(dt, dx[,dy,dz])` sets time and spatial steps before updates. Use spacing values that satisfy the selected scheme’s stability requirements."),
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


def _build_pdf(body_font):
    buf=BytesIO(); pw,ph=letter; margin=.38*inch; header=.34*inch
    usable=ph-2*margin-header
    gutter=.22*inch; col_w=(pw-2*margin-gutter)/2
    frames=[Frame(margin,margin,col_w,usable,leftPadding=5,rightPadding=5,topPadding=2,bottomPadding=2), Frame(margin+col_w+gutter,margin,col_w,usable,leftPadding=5,rightPadding=5,topPadding=2,bottomPadding=2)]
    def header_fn(canvas,doc):
        canvas.saveState(); canvas.setFont("Helvetica-Bold",16); canvas.drawString(margin,ph-margin-9,TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888")); canvas.setLineWidth(.5); canvas.line(margin,ph-margin-14,pw-margin,ph-margin-14); canvas.restoreState()
    intro=ParagraphStyle("intro",fontName="Helvetica",fontSize=body_font+.25,leading=(body_font+.25)*1.12,spaceAfter=3)
    heading=ParagraphStyle("heading",fontName="Helvetica-Bold",fontSize=body_font+1.4,leading=(body_font+1.4)*1.15,spaceBefore=3.0,spaceAfter=3.0,borderColor=colors.HexColor("#888888"),borderWidth=.55,borderPadding=3,backColor=colors.HexColor("#F7F7F7"),leftIndent=0)
    entry=ParagraphStyle("entry",fontName="Helvetica",fontSize=body_font,leading=body_font*1.10,spaceAfter=.9,leftIndent=7,firstLineIndent=-7)
    box=ParagraphStyle("box",fontName="Helvetica",fontSize=body_font,leading=body_font*1.10,spaceAfter=2.5,backColor=colors.HexColor("#F4F4F4"),borderPadding=3)
    story=[Paragraph("<b>Conventions</b><br/>" + _markup(INTRO + " " + CONVENTIONS),box)]
    for section_i,(title,entries) in enumerate(SECTIONS):
        items=[Paragraph(f"{_markup(k, heading=True)}: {_markup(d, heading=(k in ('Properties', 'Indexing')))}",entry) for k,d in entries]
        story.append(Paragraph(title,heading))
        story.extend(items)
    story.append(Paragraph(f"<b>{_markup(FOOTER)}</b>",intro))
    doc=BaseDocTemplate(buf,pagesize=letter,leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=margin,invariant=1)
    doc.addPageTemplates(PageTemplate(id="main",frames=frames,onPage=header_fn)); doc.build(story)
    return buf.getvalue(), doc.page


def _pdf():
    # Keep body text comfortably readable rather than shrinking to one page.
    data,pages=_build_pdf(9.2)
    print(f"PAL cheatsheet generated {pages} pages at 9.2 pt body font.")
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
