"""Execute the Manual quickstart as published, including its population assertion."""

import ast
from pathlib import Path


def test_manual_quickstart():
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    start = manual.index("```python\n") + len("```python\n")
    end = manual.index("\n```", start)
    snippet = manual[start:end]
    ast.parse(snippet)
    namespace = {"__name__": "pal_documentation_test"}
    exec(compile(snippet, "Documentation/MANUAL.md", "exec"), namespace)
    pop = namespace["pal"].NewPopGrid((40, 40))
    for _ in range(10):
        namespace["Step"](pop)
    assert pop.GetPop() == 16000


def test_cheatsheet_nonvisual_snippets():
    """Check the published Grid, AgentGrid, PopGrid/PDEgrid, and list examples."""
    sheet = (Path(__file__).resolve().parents[2] / "Documentation" / "CHEATSHEET.md").read_text(encoding="utf-8")
    blocks = sheet.split("```python\n")[1:]
    for block in blocks[:4]:
        snippet = block.split("\n```", 1)[0]
        ast.parse(snippet)
        namespace = {"__name__": "pal_cheatsheet_test"}
        exec("import PythonAutomataLibrary as pal\n" + snippet, namespace)
