"""Regression against the real Persian example's annotated Grid Setup kernel."""

import runpy
from pathlib import Path

import PythonAutomataLibrary as pal


def test_persian_setup_real_example_source():
    example = Path(__file__).parents[2] / "Examples" / "ComplexExamples" / "Persian.py"
    ns = runpy.run_path(str(example), run_name="pal_persian_regression")
    types = pal.NewGrid(dimensions=(-ns["DIM"], -ns["DIM"]), dtype="int8")

    ns["Setup"](types)

    center = ns["DIM"] // 2
    assert types[center, center] == ns["DEFECTOR"]
    assert sum(int(types[i]) for i in range(len(types))) == 1
