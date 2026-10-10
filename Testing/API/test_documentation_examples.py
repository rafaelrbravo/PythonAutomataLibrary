"""Execute the Manual quickstart as published, including its population assertion."""

import ast
import runpy
from pathlib import Path



def test_manual_quickstart(tmp_path):
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    start = manual.index("```python\n") + len("```python\n")
    end = manual.index("\n```", start)
    snippet = manual[start:end]
    ast.parse(snippet)
    source_path = tmp_path / "manual_quickstart.py"
    source_path.write_text(snippet, encoding="utf-8")
    namespace = runpy.run_path(str(source_path), run_name="pal_documentation_test")
    pop = namespace["pal"].NewPopGrid((40, 40))
    for _ in range(10):
        namespace["Step"](pop)
    assert pop.GetPop() == 16000
    assert pop[0] == 10
    assert pop[39, 39] == 10


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
    assert markup == "Call <b>pal.NewGrid((3, 5), float)</b> for a grid."
    assert "Courier" not in path.read_text(encoding="utf-8")


def test_cheatsheet_pdf_fits_one_page():
    """Check the compact PDF renderer still fits its target page."""
    path = Path(__file__).resolve().parents[2] / "Documentation" / "generators" / "generate_cheatsheet.py"
    namespace = runpy.run_path(str(path), run_name="pal_cheatsheet_layout_test")
    fitted = [(size, namespace["_build_pdf"](size)[1]) for size in (round(13.5-i*.1, 1) for i in range(86))]
    assert any(pages == 1 for _, pages in fitted), "No readable font size fits on one page"
    assert "FrameBreak" not in path.read_text(encoding="utf-8")

def test_manual_checkpoint_snippet():
    """Execute the checkpoint example as published in the Manual."""
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    checkpoint_section = manual.split("Checkpointing and reproducibility\n", 1)[1].split("\n## ", 1)[0]
    assert checkpoint_section.count("```python\n") == 1, "Expected one checkpoint example in the checkpointing section"
    snippet = checkpoint_section.split("```python\n", 1)[1].split("\n```", 1)[0]
    ast.parse(snippet)
    exec(compile(snippet, "pal_manual_checkpoint.py", "exec"), {"__name__": "pal_documentation_test"})


def test_manual_radius_snippet():
    """Execute the Manual's wrapped radius search exactly as published."""
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    section = manual.split("### Iterating nearby agents\n", 1)[1].split("\n## ", 1)[0]
    assert section.count("```python\n") == 1
    snippet = section.split("```python\n", 1)[1].split("\n```", 1)[0]
    ast.parse(snippet)
    exec(compile(snippet, "pal_manual_radius.py", "exec"), {"__name__": "pal_documentation_test"})


def test_manual_gradient_snippet():
    """Execute the finite-difference gradient check exactly as published."""
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    section = manual.split("### A hand-checkable field gradient\n", 1)[1].split("\n### Numerical update order", 1)[0]
    assert section.count("```python\n") == 1
    snippet = section.split("```python\n", 1)[1].split("\n```", 1)[0]
    ast.parse(snippet)
    exec(compile(snippet, "pal_manual_gradient.py", "exec"), {"__name__": "pal_documentation_test"})
