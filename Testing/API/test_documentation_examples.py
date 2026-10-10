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
    import linecache
    source_name = str(Path(__file__).resolve().parents[2] / "Documentation" / "_manual_quickstart.py")
    linecache.cache[source_name] = (len(snippet), None, [line + "\n" for line in snippet.splitlines()], source_name)
    exec(compile(snippet, source_name, "exec"), namespace)
    pop = namespace["pal"].NewPopGrid((40, 40))
    for _ in range(10):
        namespace["Step"](pop)
    assert pop.GetPop() == 16000
    assert pop[0] == 10
    assert pop[39, 39] == 10


def test_cheatsheet_nonvisual_snippets():
    """Check the published Grid, AgentGrid, PopGrid/PDEgrid, and list examples."""
    sheet = (Path(__file__).resolve().parents[2] / "Documentation" / "CHEATSHEET.md").read_text(encoding="utf-8")
    blocks = sheet.split("```python\n")[1:]
    assert len(blocks) >= 4, "Expected four nonvisual Cheatsheet examples"
    for index, block in enumerate(blocks[:4]):
        snippet = block.split("\n```", 1)[0]
        ast.parse(snippet)
        namespace = {"__name__": "pal_cheatsheet_test"}
        exec("import PythonAutomataLibrary as pal\n" + snippet, namespace)
        if index == 0:
            assert namespace["value"] == 1.0
            assert namespace["copy"].shape == (3, 5)
        elif index == 1:
            assert namespace["agents"].GetPop() == 0
        elif index == 2:
            assert namespace["pop"].GetPop() == 0
        elif index == 3:
            assert list(namespace["snapshot"]) == [4, 8]
