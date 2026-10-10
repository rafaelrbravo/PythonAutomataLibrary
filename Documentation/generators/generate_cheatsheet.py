"""Generate PAL CHEATSHEET.md and CHEATSHEET.pdf from one compact definition.

The PDF follows the BET cheat-sheet approach: content is readable in this source,
and the renderer chooses the largest body font that fits on one US Letter page.

Usage: python Documentation/generators/generate_cheatsheet.py [--check]
"""
import argparse, html, re
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph

ROOT = Path(__file__).resolve().parents[2]
MD_TARGET = ROOT / "Documentation" / "CHEATSHEET.md"
PDF_TARGET = ROOT / "Documentation" / "CHEATSHEET.pdf"
TITLE = "Python Automata Library (PAL) Cheat Sheet"
INTRO = "`import PythonAutomataLibrary as pal` · Safe mode is default. Call `pal.FastMode()` before constructing any PAL object. Put substantive model kernels under `@pal.njit(cache=True)`."
CONVENTIONS = "`dims` gives grid size per axis: `(nx,)` for 1D, `(nx, ny)` for 2D, `(nx, ny, nz)` for 3D. `x...` means `x`, `x,y`, or `x,y,z` in 1D/2D/3D. `Box(x1,x2,...)` alternates lower and exclusive upper bounds per axis. Negative dimensions enable wrapping; out-of-range coordinates wrap on wrapped axes and are skipped on nonwrapped axes."

SECTIONS = [
("Create state", [
("`pal.NewGrid(dims, dtype)`", "Create a typed lattice storing one value per site."), ("`pal.NewAgentGrid(dims, numAgentProps=0, isStackable=False)`", "Create agents with optional properties; use `dims=()` for nonspatial agents."),
("`pal.NewPopGrid(dims, capacity=None)`", "Create integer counts per site; capacity limits total population."), ("`pal.NewPDEgrid(dims)`", "Create a continuous field for PDE updates."),
("`pal.NewIList()`", "Create a mutable integer list for collecting/reusing query results."), ("`pal.NewMultinomial()`", "Create a sampler for binomial/multinomial population draws."),
("Dimensions", "1–3 axes; a negative dimension wraps that axis, e.g. `(-nx, ny)`."),]),
("Shared lattice geometry", [
("Properties", "`xDim/yDim/zDim`, `nDims`, `wrapX/Y/Z`; a negative dimension enables wrapping on that axis."),
("`ToI(x...)`", "Convert 1D/2D/3D coordinates into one linear site index."),
("`ItoX/Y/Z(i)`", "Convert a linear site index back to its axis coordinate."),
("Regions", "`Box(x1,x2,...)` selects a rectangular region; each axis has a lower bound and exclusive upper bound."),
("`pal.MooreHood(dim, excludeCenter=False)`", "Build offsets for all neighboring sites, including diagonals."),
("`pal.VonNeumannHood(dim, excludeCenter=False)`", "Build axis-aligned offsets, excluding diagonal neighbors."),
("`pal.CircleHood(dim, rad, excludeCenter=False)`", "Build offsets within a radius; hood constructors return relative offsets, not absolute indices."),
("`grid.Hood(hood, x...)`", "Apply relative offsets at a coordinate; `excludeCenter=True` omits the center when constructing the hood."),]),
("Grid / common indexing", [
("`g[i]`", "Read or write a site by linear index."), ("`g[x...]`", "Read or write a site using dimensional coordinates."), ("Slices", "Return detached NumPy copies, not live views; editing a slice does not update the grid."),
("Pattern", "`g = pal.NewGrid((40,40), float)` · `g[10,12] = 1.0` · `v = g[10,12]`; supported dtypes include bool, fixed-width integers, and float32/float64."),]),
("Draw / output", [
("`pal.StartPixWindow(...)`", "Create a pixel buffer and display window; `headless=True` supports noninteractive rendering."),
("`pix[x,y] = RGB`", "Set a pixel color in the shared pixel buffer."),
("`win.Update()`", "Refresh the displayed window after drawing."),
("`win.Save(path, block=True)`", "Save the current image; blocking waits for completion."),
("`win.Close()`", "Close the window and release display resources."),
("`StartGif(path,delay=100)`", "Begin recording an animated GIF with the requested frame delay."),
("`AddGifFrame(block=False)`", "Append the current rendered frame to the GIF."),
("`StopGif()`", "Finish the GIF; update the window before capturing frames."),
("`pal.StartOpenGLWindow(...)`", "Create an OpenGL window for drawing 2D or 3D scenes."),
("`Circle(...)` / `Box(...)` / `Line(...)`", "Draw circular, box-shaped, or line primitives in the scene."),
("`BoxSQ(...)` / `Borders(...)`", "Draw lattice-aligned boxes or grid boundaries."),
("`Camera(...)`", "Set the viewpoint for a 3D scene."),
("`Background(...)` / `Clear()`", "Set the scene background or clear the current frame."),]),("Lists + randomness", [
("IList methods", "`Append(i)` adds an integer; `Clear()` empties the list; `Random()` picks an entry; `Shuffle()` reorders entries; `All()` copies the list; `Iter()` traverses without copying."),
("`pal.Seed(seed)`", "Seed PAL’s shared random stream for reproducible call sequences."),
("`pal.Random()`", "Draw a uniform random number from PAL’s random stream."),
("`pal.RandInt(n)`", "Draw an integer from 0 through n−1; n must be positive. PAL’s stream is separate from NumPy’s RNG."),
("`m.Binomial(n,p)`", "Draw a binomial count with n trials and probability p."),
("`m.Setup(...)` / `m.Sample(...)`", "Configure a multinomial sampler, then draw counts from it."),]),
("AgentGrid", [
("`NewAgentSQ(x...)` / `MoveSQ(a,x...)`", "Create or move an agent using lattice-site coordinates; documented linear-site forms are also supported."),
("`NewAgent(x...)` / `Move(a,x...)`", "Create or move an agent using continuous coordinates rather than lattice sites."),
("`grid[a,p]`", "Read or write property p for agent a."),
("`I(a)`", "Get an agent’s linear lattice-site index."),
("`XSQ/YSQ/ZSQ(a)`", "Get integer lattice coordinates; `X/Y/Z(a)` return continuous positions."),
("`Alive(a)` / `Dispose(a)`", "Check whether an agent is still valid, or remove it from the grid."),
("`GetPop()`", "Count agents currently in the grid."),
("`AgentsAt(x...)` / `LastAgent(x...)`", "Find agents at a site or retrieve the last agent there."),
("`AgentsInRadius(...)`", "Find agents near a position; wrapped displacement respects periodic boundaries."),
("`counts[x...]`", "Read site occupancy counts without modifying them."),
("Iteration", "`for a in agents.All(): ...` iterates a detached snapshot and is safe for structural mutation such as `Dispose`. `AgentsInRadius(...)` returns nearby agents; wrapped displacements account for periodic boundaries."),
("`DispWrapX/Y/Z(p1,p2)`", "Compute the shortest displacement along a periodic axis."),]),
("PopGrid + PDEgrid", [
("`Add(v,...)`", "Accumulate pending changes without immediately changing current values."),
("`Update()`", "Apply accumulated changes together as a synchronous update."),
("`Reset()`", "Clear both current values and pending changes; direct indexed assignment changes current state immediately."),
("`pop.GetPop()`", "Return total population across all sites."),
("`pop.All()`", "Return indices of sites with nonzero population, not the population values."),
("PDE setup", "`field.SetTimeSpaceStep(dt, dx[,dy,dz])` sets time and spatial steps before updates. Use spacing values that satisfy the selected scheme’s stability requirements."),
("`Diffusion(...)`", "Diffuse the field using the standard lattice scheme."),
("`DiffusionMask(...)`", "Diffuse subject to a mask of permitted sites or interfaces."),
("`DiffusionField(...)`", "Use another field to vary diffusion across space."),
("`DiffusionInterfaces(...)`", "Specify diffusion behavior at interfaces."),
("`DiffusionADI(...)`", "Use an alternating-direction implicit diffusion scheme."),
("`DiffusionRadialCircle/Sphere(...)`", "Use radial diffusion for circular or spherical geometry."),
("`Advection(...)`", "Move field values with a specified velocity."),
("`AdvectionField(...)`", "Use a field to define spatially varying advection."),
("`AdvectionInterfaces(...)`", "Specify advection behavior at interfaces."),]),

]

FOOTER = "More detail: Manual · API Guide · API Reference (Documentation/)."


def _md():
    out=["# PAL Cheatsheet", "", "## Conventions", "", INTRO + " " + CONVENTIONS, ""]
    for title, entries in SECTIONS:
        out += [f"## {title}", ""]
        for key, desc in entries: out.append(f"- **{key}** — {desc}")
        out.append("")
    out.append(f"**{FOOTER}**")
    return "\n".join(out).rstrip()+"\n"


def _markup(s):
    s=html.escape(s, quote=False).replace("-&gt;", "→")
    s=re.sub(r"`([^`]+)`", r"<b>\1</b>", s)
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
        items=[Paragraph(f"<b>{_markup(k)}</b> — {_markup(d)}",entry) for k,d in entries]
        story.append(KeepTogether([Paragraph(title,heading)]+items))
    story.append(Paragraph(f"<b>{_markup(FOOTER)}</b>",intro))
    doc=BaseDocTemplate(buf,pagesize=letter,leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=margin,invariant=1)
    doc.addPageTemplates(PageTemplate(id="main",frames=frames,onPage=header_fn)); doc.build(story)
    return buf.getvalue(), doc.page


def _pdf():
    for i in range(51):
        size=round(13.5-i*.1,1)
        if size < 5.0: break
        data,pages=_build_pdf(size)
        if pages==1:
            print(f"PAL cheatsheet fits one page at {size:.1f} pt body font.")
            return data
    raise RuntimeError("PAL cheatsheet does not fit one page at 5.0 pt; consolidate content further.")


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
