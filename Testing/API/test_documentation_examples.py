"""Execute current API Guide tutorial snippets against PAL, including compiled paths."""

import ast
import re
import runpy
from pathlib import Path

import pytest


GUIDE = Path(__file__).resolve().parents[2] / "Documentation" / "API_GUIDE.md"


def _blocks():
    guide = GUIDE.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)\n```", guide, flags=re.S)


def _namespace(code):
    namespace = {"__name__": "pal_api_guide_test"}
    exec(compile(code, str(GUIDE), "exec"), namespace)
    return namespace


def test_all_python_blocks_parse():
    for number, block in enumerate(_blocks(), 1):
        # A few blocks intentionally show indented replacements for a function body.
        ast.parse("if True:\n" + "\n".join("    " + line for line in block.splitlines()), filename=f"block {number}")


def test_tutorial_initial_cells():
    ns = _namespace(_blocks()[0])
    cells = ns["Setup"]()
    assert len(cells.All()) == 4
    assert all(cells.counts[x, y] == 1 for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)))


def test_tutorial_agent_movement_and_compilation():
    blocks = _blocks()
    ns = _namespace(blocks[1])
    exec(blocks[2], ns)
    cells = ns["Setup"]()
    pal = ns["pal"]
    ns["MOORE"] = pal.MooreHood(2, excludeCenter=True)
    empty = pal.NewIList()
    neighbors = ns["EmptyNeighbors"](cells, cells.All()[0], empty).All()
    assert len(neighbors) > 0
    exec(blocks[4], ns)
    for _ in range(3):
        ns["Step"](cells, empty)
        assert len(cells.All()) == 4
    exec(blocks[7], ns)
    exec(blocks[6], ns)
    for _ in range(3):
        ns["Step"](cells, empty)
        assert len(cells.All()) == 4
    exec(blocks[9], ns)
    nutrient = pal.NewPDEgrid((ns["SIZE"], ns["SIZE"]))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    ns["StepNutrient"](cells, nutrient)
    assert nutrient[14, 14] <= 1.0


def test_tutorial_population_compiled():
    ns = _namespace(_blocks()[11])
    pal = ns["pal"]
    pal.Seed(1234)
    cells = pal.NewPopGrid((ns["SIZE"], ns["SIZE"]))
    for x, y in ((14, 14), (15, 14), (14, 15), (15, 15)):
        cells[x, y] = 1
    multi = pal.NewMultinomial()
    nutrient = pal.NewPDEgrid((ns["SIZE"], ns["SIZE"]))
    nutrient.SetTimeSpaceStep(1.0, 1.0, 1.0)
    nutrient[:] = 1.0
    for _ in range(5):
        ns["Step"](cells, multi)
        assert cells.GetPop() == 4
        ns["StepNutrient"](cells, nutrient)


def _cheatsheet_sections():
    path = Path(__file__).resolve().parents[2] / "Documentation" / "generators" / "generate_cheatsheet.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SECTIONS" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("SECTIONS assignment not found")


def test_cheatsheet_core_api_coverage():
    """Ensure the compact cheatsheet retains the main PAL API families."""
    sections = _cheatsheet_sections()
    source = repr(sections)
    for term in (
        "NewGrid", "NewAgentGrid", "NewPopGrid", "NewPDEgrid", "ToI",
        "MooreHood", "StartPixWindow", "StartOpenGLWindow", "NewIList",
        "RandInt", "NewAgentSQ", "AgentsInRadius", "DiffusionADI", "Advection",
    ):
        assert term in source, f"Missing core API term: {term}"
    assert len(sections) >= 6


def test_cheatsheet_markup_bolds_syntax_without_monospace():
    """Keep API tokens bold in sans-serif markup, not rendered in Courier."""
    path = Path(__file__).resolve().parents[2] / "Documentation" / "generators" / "generate_cheatsheet.py"
    namespace = runpy.run_path(str(path), run_name="pal_cheatsheet_markup_test")
    markup = namespace["_markup"]("Call `pal.NewGrid((3, 5), float)` for a grid.")
    assert markup == "Call <b>pal.NewGrid</b>((3, 5), float) for a grid."
    assert "Courier" not in path.read_text(encoding="utf-8")


def test_cheatsheet_pdf_fits_one_page():
    """The curated cheatsheet must fit in its published one-page layout."""
    path = Path(__file__).resolve().parents[2] / "Documentation" / "generators" / "generate_cheatsheet.py"
    namespace = runpy.run_path(str(path), run_name="pal_cheatsheet_layout_test")
    _, pages = namespace["_build_pdf"]()
    assert pages == 1


