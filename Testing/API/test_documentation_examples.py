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
