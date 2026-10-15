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


def _run_complete_program(tmp_path, block_number, replacements=None):
    """Run a published complete program from a real source file."""
    code = _blocks()[block_number]
    # The GUI tutorial waits for manual window closure; replace its event loop
    # with a bounded headless smoke test while retaining the published functions.
    if replacements:
        for old, new in replacements:
            assert old in code, f"Expected tutorial fragment missing: {old!r}"
            code = code.replace(old, new)
    source = tmp_path / f"api_guide_program_{block_number}.py"
    source.write_text(code, encoding="utf-8")
    runpy.run_path(str(source), run_name="__main__")


def test_complete_initial_cells(tmp_path):
    _run_complete_program(tmp_path, 0)


def test_complete_visualization(tmp_path, monkeypatch):
    import PythonAutomataLibrary as pal

    class FakeWindow:
        def __init__(self):
            self.updates = 0
        def Update(self):
            self.updates += 1
        def IsOpen(self):
            return self.updates < 2
        def Close(self):
            pass

    class FakePixels:
        def __setitem__(self, key, value):
            pass

    window = FakeWindow()
    monkeypatch.setattr(pal, "StartPixWindow", lambda *args, **kwargs: (FakePixels(), window))
    monkeypatch.setattr(pal, "AwaitWindows", lambda: None)
    _run_complete_program(tmp_path, 1)
    assert window.updates >= 1


def test_complete_population_random_walk(tmp_path, monkeypatch):
    import PythonAutomataLibrary as pal

    class FakeWindow:
        def __init__(self):
            self.updates = 0
        def Update(self):
            self.updates += 1
        def IsOpen(self):
            return self.updates < 3
        def Close(self):
            pass

    class FakePixels:
        def __setitem__(self, key, value):
            pass

    window = FakeWindow()
    monkeypatch.setattr(pal, "StartPixWindow", lambda *args, **kwargs: (FakePixels(), window))
    monkeypatch.setattr(pal, "AwaitWindows", lambda: None)
    _run_complete_program(tmp_path, 11)
    assert window.updates == 3


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


