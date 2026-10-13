"""Generate PAL CHEATSHEET.md and CHEATSHEET.pdf from one compact definition.

The one-page PDF and Markdown use the same independently curated entries.

Usage: python Documentation/generators/generate_cheatsheet.py [--check]

The generated Markdown and PDF must be refreshed together after API changes.
"""
import argparse, html, re
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
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
("START / UTILITIES", [
("pal.FastMode() · pal.njit(cache=True)", "Select fast mode before object creation; compile model kernels."),
("pal.Seed(s) · Random() · RandInt(n)", "Shared PAL RNG; seed for reproducibility."),
("pal.NewGrid(dims,dtype) · NewAgentGrid(dims,props=0,stack=False)", "Typed sites; mobile agents."),
("pal.NewPopGrid(dims,capacity=None) · NewPDEgrid(dims)", "Integer populations; continuous fields."),
("pal.NewIList() · NewMultinomial(other=None)", "Reusable integer list; random sampler."),
("pal.MooreHood(dim,excludeCenter=False) · VonNeumannHood(...) · CircleHood(dim,rad,...)", "Relative neighbor offsets."),
("pal.RGB(r,g,b) · GetR/G/B(c) · ColorScale(colors,value)", "Pack, extract, interpolate colors."),
("pal.AwaitWindows(timeout=30)", "Wait for rendering windows."),
]),
("SHARED GRID API", [
("dims=(nx,ny,nz); negative size = wrap", "1–3 dimensions; unused axes omitted."),
("grid.xDim... · nDims · wrapX...", "Axis sizes, dimensionality, periodicity."),
("len(grid) · grid[i] · grid[x,y,z] · grid[...] = value", "Site count; read/write; slices copy."),
("grid.ToI(x,y,z) · ItoX...(i)", "Coordinate ↔ linear index."),
("grid.InWrapX...(x)", "Wrap coordinate on periodic axis."),
("grid.Box(x1,x2,y1,y2,z1,z2)", "Half-open rectangular iterator."),
("grid.Hood(hood,x,y,z,unroll=False)", "Offset neighborhood at a site."),
]),
("AGENT GRID", [
("a=agents.NewAgentSQ(x,y,z) · NewAgent(x,y,z)", "Create at lattice site / continuous point."),
("agents.MoveSQ(a,x,y,z) · Move(a,x,y,z)", "Move by lattice / continuous position."),
("agents.I(a) · XSQ...(a) · X...(a)", "Site index, lattice and real coordinates."),
("agents.Alive(a) · Dispose(a) · GetPop()", "Life status, removal, count."),
("agents[a,p] · agents.counts", "Agent properties; read-only occupancy."),
("agents.AgentsAt(x,y,z) · LastAgent(x,y,z)", "Site occupants / last occupant."),
("agents.AgentsInRadius(rad,x,y,z,exclude=None)", "Nearby agents; periodic-aware."),
("agents.All(shuffle=False)", "Snapshot of living agent IDs."),
("agents.InWrapSQX...(x) · DispWrapX...(p1,p2)", "Wrap lattice coordinate; shortest displacement."),
]),
("POP GRID / PDE FIELD", [
("pop.Add(v,x,y,z) · Update() · Reset()", "Queue count deltas, apply, clear."),
("pop.GetPop() · All()", "Total population; occupied indices."),
("field.Add(v,x,y,z) · Update() · Reset()", "Queue field deltas, apply, clear."),
("field.SetTimeSpaceStep(dt,dx,dy=1,dz=1) · Dt() · Dx...()", "Numerical spacing."),
("field.Diffusion(rate,BC...) · DiffusionMask(rate,mask,...)", "Uniform / masked diffusion."),
("field.DiffusionField(rates,BC...) · DiffusionInterfaces(rx,ry,rz,BC...)", "Cell / face coefficients."),
("field.DiffusionADI(rate,BC...) · DiffusionRadialCircle/Sphere(rate,outerBC)", "Implicit / radial diffusion."),
("field.Advection(vx,vy,vz,BC...) · AdvectionField(vx,vy,vz,BC...)", "Uniform / spatial velocity."),
("field.AdvectionInterfaces(vx,vy,vz,BC...)", "Face-centered velocity."),
("BC... = xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC", "Optional face boundary conditions."),
]),
("LISTS / SAMPLING", [
("lst.Append(i) · Clear() · Random() · Shuffle()", "Integer-list operations."),
("lst.All() · Iter() · len(lst) · lst[i]", "Copy / no-copy traversal, indexing."),
("m.Binomial(n,p) · Setup(n) · Sample(p)", "Binomial sampling."),
]),
("SHARED WINDOW API", [
("win.IsOpen() · Update() · Save(path,block=False) · Close()", "Window lifecycle / image output."),
("win.StartGif(path,delay=100) · AddGifFrame(block=False) · StopGif()", "GIF recording; OpenGL supports timeout options."),
]),
("PIX WINDOW", [
("pix,win=pal.StartPixWindow(nx,ny,scale=1,title='PAL',headless=False)", "Pixel renderer."),
("pix[x,y]=pal.RGB(r,g,b) · pix[i]=color", "Draw packed RGB pixels."),
("pix.xDim · yDim · ToI(x,y) · ItoX/Y(i) · len(pix)", "Buffer geometry."),
]),
("OPENGL WINDOW / DRAW", [
("draw,win=pal.StartOpenGLWindow(nx,ny,nz=None,width=800,height=800,...)", "2D/3D primitive renderer."),
("draw.Circle(rad,color,x,y,z=0) · Box(xLen,color,x,y,z=0,...)", "Add circles/spheres and boxes."),
("draw.BoxSQ(color,x,y,z=0) · Line(width,color,x1,y1,x2,y2,...)", "Site boxes and segments."),
("draw.Clear() · win.Clear() · win.Background(color)", "Reset primitives / scene / background."),
("win.Borders(width,color) · win.Camera(x,y,z,yaw=None,pitch=None)", "Domain border and 3D camera."),
]),
]

def _api_entries():
    """Return the independently curated cheatsheet content."""
    return SECTIONS


def _md():
    lines = ["# PAL Cheatsheet", "", INTRO, "",
             "Axis shorthand: `...` after an X/x-specific API name denotes corresponding Y/y and Z/z variants.", "",
             "Curated one-page API map; see API Guide and API Reference for details.", ""]
    for title, rows in _api_entries():
        lines += ["## " + title, ""]
        for sig, description in rows:
            lines.append("- `" + sig + "`: " + description)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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


def _build_pdf():
    """Render a one-page portrait, two-column quick reference."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import portrait
    from reportlab.pdfbase.pdfmetrics import stringWidth
    buf = BytesIO()
    pw, ph = portrait(letter)
    c = canvas.Canvas(buf, pagesize=(pw, ph), pageCompression=1, invariant=1)
    c.setTitle(TITLE)
    margin, gap, top, bottom = 17, 12, 40, 17
    ncols = 2
    width = (pw - 2 * margin - gap * (ncols - 1)) / ncols
    rows = [(title, sig.strip("&#96;"), desc) for title, entries in SECTIONS for sig, desc in entries]
    def split_lines(value, font, size, max_width):
        words = value.split()
        lines, line = [], ""
        for word in words:
            candidate = (line + " " + word).strip()
            if line and stringWidth(candidate, font, size) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines
    def layout(size):
        lead = size * 1.12
        heading_h = size * 1.45 + 4
        blocks = []
        for title, entries in SECTIONS:
            pieces = []
            for sig, desc in entries:
                sig = sig.strip("&#96;")
                lines = split_lines(sig, "Helvetica-Bold", size, width - 8)
                details = split_lines(desc, "Helvetica", size - .4, width - 8)
                pieces.append((lines, details, lead * (len(lines) + len(details)) + 2))
            blocks.append((title, pieces, heading_h + sum(p[2] for p in pieces) + 4))
        columns = [[], []]
        heights = [0, 0]
        usable = ph - top - bottom
        for title, pieces, h in blocks:
            col = min(range(ncols), key=lambda i: heights[i])
            if heights[col] + h > usable:
                candidates = [i for i in range(3) if heights[i] + h <= usable]
                if not candidates:
                    return None
                col = min(candidates, key=lambda i: heights[i])
            columns[col].append((title, pieces))
            heights[col] += h
        return columns, heights, lead, heading_h
    selected = None
    for size in (8.0, 7.8, 7.6, 7.4, 7.2, 7.0, 6.8, 6.6, 6.4, 6.2):
        selected = layout(size)
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("Curated cheatsheet cannot fit on one page at minimum readable font")
    columns, heights, lead, heading_h = selected
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, ph - 22, TITLE)
    c.setFont("Helvetica", 7.5)
    c.drawRightString(pw - margin, ph - 22, "1-page API map · X... includes Y/Z variants")
    c.setStrokeColor(colors.HexColor("#666666"))
    c.line(margin, ph - 28, pw - margin, ph - 28)
    for col, blocks in enumerate(columns):
        x = margin + col * (width + gap)
        y = ph - top
        for title, pieces in blocks:
            c.setFillColor(colors.HexColor("#E9EDF2"))
            c.rect(x, y - heading_h + 2, width, heading_h - 2, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#172B4D"))
            c.setFont("Helvetica-Bold", size + .6)
            c.drawString(x + 3, y - size - 2, title)
            y -= heading_h
            for sig_lines, desc_lines, _ in pieces:
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", size)
                for line in sig_lines:
                    c.drawString(x + 3, y - size, line)
                    y -= lead
                c.setFillColor(colors.HexColor("#414141"))
                c.setFont("Helvetica", size - .4)
                for line in desc_lines:
                    c.drawString(x + 3, y - size, line)
                    y -= lead
                y -= 2
            y -= 4
    c.setFont("Helvetica", 6.7)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(margin, 8, "dims=(nx,ny,nz); negative dimension wraps · API Reference has exact signatures and defaults")
    c.save()
    print(f"One-page cheatsheet: {sum(len(e) for _,e in SECTIONS)} curated entries, {size:.1f} pt, column heights {[round(h) for h in heights]}")
    return buf.getvalue(), 1


def _pdf():
    data, pages = _build_pdf()
    if pages != 1:
        raise RuntimeError(f"Cheatsheet must be one page, got {pages}")
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
