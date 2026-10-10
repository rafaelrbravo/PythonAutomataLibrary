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
    assert len(blocks) == 5, "Expected four nonvisual examples and one drawing example"
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


def test_cheatsheet_draw_headless(tmp_path):
    """Run the published drawing example off-screen and inspect its saved pixel."""
    import numpy as np
    import pytest
    Image = pytest.importorskip("PIL.Image")

    sheet = (Path(__file__).resolve().parents[2] / "Documentation" / "CHEATSHEET.md").read_text(encoding="utf-8")
    blocks = sheet.split("```python\n")[1:]
    assert len(blocks) == 5
    snippet = blocks[4].split("\n```", 1)[0]
    ast.parse(snippet)
    snippet = snippet.replace("pal.StartPixWindow(40, 40, scale=4)", "pal.StartPixWindow(40, 40, scale=4, headless=True)")
    snippet = snippet.replace('"frame.png"', "str(output)")
    output = tmp_path / "frame.png"
    namespace = {"output": output}
    exec("import PythonAutomataLibrary as pal\n" + snippet, namespace)
    assert output.exists()
    image = np.asarray(Image.open(output).convert("RGB"))
    red = np.all(image == [255, 0, 0], axis=2)
    assert red.sum() == 16, "Expected one 4x4 scaled red pixel"
    # Headless output transposes x/y and reverses the y axis before scaling.
    expected_y = (40 - 1 - 12) * 4
    expected_x = 10 * 4
    assert red[expected_y:expected_y + 4, expected_x:expected_x + 4].all()
    assert np.all(image[~red] == 0), "Expected all other pixels to remain black"


def test_manual_checkpoint_snippet():
    """Execute the checkpoint example as published in the Manual."""
    manual = (Path(__file__).resolve().parents[2] / "Documentation" / "MANUAL.md").read_text(encoding="utf-8")
    blocks = manual.split("```python\n")[1:]
    assert len(blocks) == 2, "Expected quickstart and checkpoint examples"
    snippet = blocks[1].split("\n```", 1)[0]
    ast.parse(snippet)
    exec(compile(snippet, "pal_manual_checkpoint.py", "exec"), {"__name__": "pal_documentation_test"})
