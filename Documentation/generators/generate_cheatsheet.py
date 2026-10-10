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
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, KeepTogether, PageTemplate, Paragraph

ROOT = Path(__file__).resolve().parents[2]
MD_TARGET = ROOT / "Documentation" / "CHEATSHEET.md"
PDF_TARGET = ROOT / "Documentation" / "CHEATSHEET.pdf"
TITLE = "Python Automata Library (PAL) Cheat Sheet"
INTRO = "`import PythonAutomataLibrary as pal` · Safe mode is default. Call `pal.FastMode()` before constructing any PAL object. Put substantive model kernels under `@pal.njit(cache=True)`."

SECTIONS = [
("Create state", [
("`pal.NewGrid(dims, dtype)`", "typed lattice"), ("`pal.NewAgentGrid(dims, numAgentProps=0, isStackable=False)`", "individual agents; `dims=()` is nonspatial"),
("`pal.NewPopGrid(dims, capacity=None)`", "integer population counts"), ("`pal.NewPDEgrid(dims)`", "continuous field"),
("`pal.NewIList()` · `pal.NewMultinomial()`", "integer query list · random-count sampler"),
("Dimensions", "1–3 axes; a negative dimension wraps that axis, e.g. `(-nx, ny)`."),]),
("Shared lattice geometry", [
("Properties", "`xDim/yDim/zDim`, `nDims`, `wrapX/Y/Z`"),
("Index conversion", "`ToI(x[,y,z])`; `ItoX/Y/Z(i)`"),
("Regions", "`Box(lo..., hi...)` uses half-open bounds; `Hood(hood, x[,y,z])` maps relative offsets to coordinates."),
("Neighborhoods", "`pal.MooreHood(dim, includeOrigin)` · `pal.VonNeumannHood(dim)` · `pal.CircleHood(dim, rad)`"),]),
("Grid / common indexing", [
("Read/write", "`g[x,y]`, `g[x,y]=v`; slices return detached NumPy copies."),
("Pattern", "`g = pal.NewGrid((40,40), float)` · `g[10,12] = 1.0` · `v = g[10,12]`"),]),
("Draw / output", [
("Pixels", "`pix, win = pal.StartPixWindow(xDim,yDim,scale=1,title='PAL',headless=False)`; set `pix[x,y]=RGB`; `win.Update()`; `win.Save(path, block=True)`; `win.Close()`."),
("GIF", "`StartGif(path,delay=100)` · `AddGifFrame(block=False)` · `StopGif()`; call `Update()` before capture."),
("OpenGL", "`pal.StartOpenGLWindow(...)`; draw with `Circle`, `Box`, `BoxSQ`, `Line`, `Borders`; scene controls include `Camera`, `Background`, `Clear`."),]),("Lists + randomness", [
("IList", "`Append(i)`, `Clear()`, `Random()`, `Shuffle()`, indexing/`len`; `All()` detached copy; `Iter()` no-copy iteration."),
("RNG", "`pal.Seed(seed)` · `pal.Random()` · `pal.RandInt(n)` → `0..n-1`; use PAL RNG for shared-stream reproducibility."),
("Multinomial", "`m.Binomial(n,p)`; `Setup(...)` then `Sample(...)` for repeated multinomial draws."),]),
("AgentGrid", [
("Create / move", "`NewAgentSQ(x,y)` / `MoveSQ(a,x,y)` for lattice positions; `NewAgent(x,y)` / `Move(a,x,y)` for continuous positions."),
("Agent state", "`grid[a,p]` property; `I(a)`, `XSQ/YSQ/ ZSQ(a)` lattice; `X/Y/Z(a)` continuous; `Alive(a)`, `Dispose(a)`."),
("Queries", "`GetPop()`, `AgentsAt(...)`, `LastAgent(...)`, `AgentsInRadius(...)`, `counts[...]`."),
("Iteration", "`for a in agents.All(): ...` is a snapshot and is safe for structural mutation such as `Dispose`."),
("Wrapping", "`DispWrapX/Y/Z(p1,p2)` gives wrapped displacement."),]),
("PopGrid + PDEgrid", [
("Transactional update", "`Add(v, ...)` changes pending state; `Update()` applies it simultaneously; `Reset()` clears current + pending. Direct `grid[...] = v` changes current state immediately."),
("Population", "`pop.GetPop()` total; `pop.All()` nonzero site indices. `capacity` limits total population."),
("PDE setup", "`field.SetTimeSpaceStep(dt, dx[,dy,dz])`"),
("Diffusion", "`Diffusion`, `DiffusionMask`, `DiffusionField`, `DiffusionInterfaces`, `DiffusionADI`; radial 1D: `DiffusionRadialCircle/Sphere`."),
("Advection", "`Advection`, `AdvectionField`, `AdvectionInterfaces`."),]),

]

FOOTER = "More detail: Manual · API Guide · API Reference (Documentation/)."


def _md():
    out=["# PAL Cheatsheet", "", INTRO, ""]
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
    buf=BytesIO(); pw,ph=letter; margin=.32*inch; header=.34*inch; gap=.12*inch
    colw=(pw-2*margin-gap)/2; usable=ph-2*margin-header
    frames=[Frame(margin,margin,colw,usable,leftPadding=3,rightPadding=3,topPadding=1,bottomPadding=1),
            Frame(margin+colw+gap,margin,colw,usable,leftPadding=3,rightPadding=3,topPadding=1,bottomPadding=1)]
    def header_fn(canvas,doc):
        canvas.saveState(); canvas.setFont("Helvetica-Bold",16); canvas.drawString(margin,ph-margin-9,TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888")); canvas.setLineWidth(.5); canvas.line(margin,ph-margin-14,pw-margin,ph-margin-14); canvas.restoreState()
    intro=ParagraphStyle("intro",fontName="Helvetica",fontSize=body_font+.25,leading=(body_font+.25)*1.12,spaceAfter=3)
    heading=ParagraphStyle("heading",fontName="Helvetica-Bold",fontSize=body_font+1.8,leading=(body_font+1.8)*1.05,spaceBefore=2.2,spaceAfter=1.4)
    entry=ParagraphStyle("entry",fontName="Helvetica",fontSize=body_font,leading=body_font*1.10,spaceAfter=.9,leftIndent=7,firstLineIndent=-7)
    box=ParagraphStyle("box",fontName="Helvetica",fontSize=body_font,leading=body_font*1.10,spaceAfter=2.5,backColor=colors.HexColor("#F4F4F4"),borderPadding=3)
    story=[Paragraph(_markup(INTRO),box)]
    for section_i,(title,entries) in enumerate(SECTIONS):
        if section_i == 4: story.append(FrameBreak())
        items=[Paragraph(f"<b>{_markup(k)}</b> — {_markup(d)}",entry) for k,d in entries]
        story.append(KeepTogether([Paragraph(title,heading)]+items))
    story.append(Paragraph(f"<b>{_markup(FOOTER)}</b>",intro))
    doc=BaseDocTemplate(buf,pagesize=letter,leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=margin)
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
