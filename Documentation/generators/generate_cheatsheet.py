"""Generate PAL CHEATSHEET.md and CHEATSHEET.pdf from one compact definition.

The PDF uses a readable two-column, full-height two-page portrait layout generated from the same
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


def _api_entries():
    """Read the generated API Reference and keep its public signatures as the source of truth."""
    reference = (ROOT / "Documentation" / "API_REFERENCE.md").read_text(encoding="utf-8")
    groups = []
    title, rows = None, []
    for line in reference.splitlines():
        if line.startswith("## "):
            if title and rows:
                groups.append((title, rows))
            title, rows = line[3:], []
        elif line.startswith("| ") and "`" in line:
            cells = [part.strip() for part in line.strip().strip("|").split("|")]
            signature = next((c.strip("`") for c in cells[:-1] if c.startswith("`")), None)
            if signature and len(cells) >= 2:
                rows.append((signature, cells[-1].replace("\\\\|", "|")))
    if title and rows:
        groups.append((title, rows))
    if not groups:
        raise RuntimeError("API Reference has no parseable declarations")
    # Share a declaration whenever two or more object APIs expose the same
    # signature. Keep factories/utilities separate from instance methods.
    object_names = ("AgentGrid", "Grid", "PopGrid", "PDEgrid",
                    "PixWindow visualization", "OpenGLWindow visualization",
                    "Pix drawing buffer", "OpenGLDraw primitive collector")
    objects = {name: rows for name, rows in groups if name in object_names}

    def normalize(sig):
        sig = sig.replace("(self, ", "(").replace("(self)", "()")
        # Hide OpenGL-only timeout arguments in the shared GIF API.
        if sig.startswith("AddGifFrame(") or sig.startswith("StopGif("):
            sig = sig.replace(", timeout=30", "").replace("(timeout=30)", "()")
        pix_shared = {
            "xDim()": "xDim() -> int",
            "yDim()": "yDim() -> int",
            "ToI(x, y)": "ToI(x: int, y: int=-1, z: int=-1) -> int",
            "ItoX(i)": "ItoX(i: int) -> int",
            "ItoY(i)": "ItoY(i: int) -> int",
            "__len__()": "__len__() -> int",
        }
        return pix_shared.get(sig, sig)

    # Place creation functions beside the objects they construct.
    factory_sections = {
        "NewIList": "IList", "NewMultinomial": "Multinomial",
        "NewGrid": "Grid", "NewAgentGrid": "AgentGrid",
        "NewPopGrid": "PopGrid", "NewPDEgrid": "PDEgrid",
        "StartPixWindow": "PixWindow visualization",
        "StartOpenGLWindow": "OpenGLWindow visualization",
    }
    relocated = {name: [] for name in factory_sections.values()}
    regrouped = []
    for title, rows in groups:
        retained = []
        for sig, desc in rows:
            function = sig.split("(", 1)[0]
            if title == "Factories and utilities" and function in factory_sections:
                relocated[factory_sections[function]].append((sig, desc))
            else:
                retained.append((sig, desc))
        regrouped.append((title, retained))
    # Window factories are documented as standalone code blocks in the reference.
    relocated["PixWindow visualization"].insert(0, (
        "StartPixWindow(xDim, yDim, scale=1, title='PAL', headless=False)",
        "Create a pixel buffer and display window."))
    relocated["OpenGLWindow visualization"].insert(0, (
        "StartOpenGLWindow(xDim, yDim, zDim=None, width=800, height=800, title='PAL', headless=False)",
        "Create an OpenGL drawing collector and window."))
    groups = [(title, relocated.get(title, []) + rows) for title, rows in regrouped]

    occurrences = {}
    for name in object_names:
        for sig, desc in objects.get(name, []):
            occurrences.setdefault(normalize(sig), []).append((name, desc))

    shared = []
    for sig, variants in occurrences.items():
        if len({name for name, _ in variants}) < 2:
            continue
        owners = ", ".join(name.replace(" visualization", "").replace(" drawing buffer", "")
                           .replace(" primitive collector", "") for name, _ in variants)
        descriptions = {desc for _, desc in variants}
        desc = (variants[0][1] if len(descriptions) == 1
                else "Same signature; consult API Reference for type-specific behavior.")
        shared.append((sig, f"[{owners}] {desc}"))

    consolidated = []
    inserted = False
    for title, rows in groups:
        if title in object_names:
            if not inserted:
                consolidated.append(("Shared object API (2+ types)", shared))
                inserted = True
            rows = [(sig, desc) for sig, desc in rows
                    if len({name for name, _ in occurrences.get(normalize(sig), [])}) < 2]
        if rows:
            consolidated.append((title, rows))

    # Compact complete axis families, including properties and agent coordinates.
    # Apply after shared-API relocation so each section is consolidated independently.
    import re
    def compact_axes(rows):
        names = {sig.split("(", 1)[0]: (sig, desc) for sig, desc in rows}
        skip = set()
        result = []
        for sig, desc in rows:
            name = sig.split("(", 1)[0]
            if name in skip:
                continue
            match = re.fullmatch(r"(.*?)(X|x)(SQ|Dim|)", name)
            if match:
                prefix, axis, suffix = match.groups()
                axes = ("X", "Y", "Z") if axis == "X" else ("x", "y", "z")
                family = [prefix + a + suffix for a in axes]
                if all(n in names for n in family):
                    tails = [names[n][0].split("(", 1)[1] for n in family]
                    if len(set(tails)) == 1:
                        skip.update(family[1:])
                        result.append((sig.replace(name + "(", name + "...(", 1),
                                       desc.rstrip(".") + " (Y/Z equivalents)."))
                        continue
            result.append((sig, desc))
        return result

    consolidated = [(title, compact_axes(rows)) for title, rows in consolidated]
    return consolidated


def _md():
    lines = ["# PAL Cheatsheet", "", INTRO, "",
             "Axis shorthand: `...` after an X/x-specific API name denotes corresponding Y/y and Z/z variants.", "",
             "Condensed from the generated API Reference; see API Guide for examples.", ""]
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


def _build_pdf(body_font=8.0):
    """Pack as many complete source entries as fit into two balanced columns."""
    buf = BytesIO()
    pw, ph = letter
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

    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=9,
                             leading=10.2, spaceBefore=3, spaceAfter=2,
                             backColor=colors.HexColor("#F1F1F1"), borderPadding=2)
    entry = ParagraphStyle("entry", fontName="Helvetica", fontSize=body_font,
                           leading=body_font * 1.13, spaceAfter=1.2,
                           leftIndent=5, firstLineIndent=-5)
    intro = ParagraphStyle("intro", fontName="Helvetica", fontSize=body_font,
                           leading=body_font * 1.13, spaceAfter=3)

    # Four columns across two portrait pages. Every public declaration is mandatory.
    max_height = usable - 12
    inner_width = col_w - 8
    groups = _api_entries()
    entries = [(title, sig, desc) for title, rows in groups for sig, desc in rows]
    total = len(entries)

    def cost(p):
        _, h = p.wrap(inner_width, max_height)
        return h + p.style.spaceBefore + p.style.spaceAfter

    def compact(sig, desc, limit):
        sig = " ".join(sig.split())
        sig = sig.replace("(self, ", "(").replace("(self)", "()")
        desc = " ".join(desc.replace("`", "").split())
        if len(desc) > limit:
            desc = desc[:limit - 3].rsplit(" ", 1)[0] + "..."
        return Paragraph("<b>" + html.escape(sig) + "</b> — " + html.escape(desc), entry)

    def pack(desc_limit):
        columns = [[], [], [], []]
        heights = [0.] * 4
        intro_p = Paragraph(_markup(INTRO), intro)
        columns[0].append(intro_p)
        heights[0] += cost(intro_p)
        current = 0
        last_title = [None] * 4
        for title, sig, desc in entries:
            p = compact(sig, desc, desc_limit)
            heading_p = Paragraph(html.escape(title), heading)
            needed = cost(p) + (cost(heading_p) if title != last_title[current] else 0)
            if heights[current] + needed > max_height:
                current += 1
                if current >= 4:
                    return None
                needed = cost(p) + cost(heading_p)
            if heights[current] + needed > max_height:
                return None
            if title != last_title[current]:
                columns[current].append(heading_p)
                heights[current] += cost(heading_p)
                last_title[current] = title
            columns[current].append(p)
            heights[current] += cost(p)
        return columns, heights

    chosen = None
    for limit in (180, 150, 120, 100, 80, 60, 40):
        chosen = pack(limit)
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError(f"Full API ({total} declarations) cannot fit on two pages at 8 pt")
    columns, heights = chosen
    print(f"PAL cheatsheet includes all {total} API declarations on two pages; "
          f"description limit {limit}; column heights {[round(h, 1) for h in heights]}.")
    story = []
    for i, column in enumerate(columns):
        if i:
            story.append(FrameBreak())
        story.extend(column)
    doc = BaseDocTemplate(buf, pagesize=letter, leftMargin=margin,
                          rightMargin=margin, topMargin=margin, bottomMargin=margin,
                          invariant=1)
    doc.addPageTemplates(PageTemplate(id="main", frames=frames, onPage=header_fn))
    doc.build(story)
    return buf.getvalue(), doc.page


def _pdf():
    data, pages = _build_pdf(8.0)
    if pages != 2:
        raise RuntimeError(f"Cheatsheet must be exactly two pages; generated {pages}.")
    print("PAL cheatsheet generated 2 portrait pages, 2 full-height columns at 8 pt body font.")
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
