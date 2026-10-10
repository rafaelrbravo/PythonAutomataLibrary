"""Regression against the real Persian example's annotated Grid Setup kernel."""

import importlib.util
from pathlib import Path

import PythonAutomataLibrary as pal


def test_persian_setup_real_example_source():
    example = Path(__file__).parents[2] / "Examples" / "ComplexExamples" / "Persian.py"
    spec = importlib.util.spec_from_file_location("pal_persian_regression", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    types = pal.NewGrid(dimensions=(-module.DIM, -module.DIM), dtype="int8")

    module.Setup(types)

    center = module.DIM // 2
    assert types[center, center] == module.DEFECTOR
    assert sum(int(types[i]) for i in range(len(types))) == 1
